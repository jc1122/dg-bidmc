#!/usr/bin/env python3
"""P95: Patient-Aware Domain Adversarial Training.

Hypothesis: A gradient reversal layer (GRL) that forces the GNN encoder to produce
patient-invariant representations will reduce the val->test generalization gap.

Approach:
  - Add a domain discriminator head that classifies which patient a graph comes from
  - Use GRL to reverse gradients during backprop so encoder *cannot* encode patient identity
  - The boundary head still receives useful boundary features
  - Lambda controls adversarial strength: 0 = baseline, higher = more domain-invariant

Configs (5 x 3 seeds = 15 trials):
  baseline  -- no adversarial loss (lambda=0, control)
  grl_001   -- max_lambda=0.01 (gentle)
  grl_005   -- max_lambda=0.05 (moderate)
  grl_01    -- max_lambda=0.1  (strong)
  grl_05    -- max_lambda=0.5  (aggressive)
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from graph_features import (
    cache_split_graphs,
    compute_feature_dims,
    DEFAULT_FEATURE_CONFIG,
)
from train import (
    load_cached_graphs,
    evaluate,
    optimize_threshold,
    nms_1d,
    build_model,
    drop_edges,
)
from torch_geometric.loader import DataLoader as PyGLoader

SEEDS = [42, 123, 456]
ROBUST_PP = {"threshold": 0.15, "nms_dist": 250}


# ===========================================================================
# Gradient Reversal Layer
# ===========================================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GRL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)


# ===========================================================================
# Domain Adversarial Model Wrapper
# ===========================================================================

class DomainAdversarialModel(nn.Module):
    """Wraps the existing DGGNN with a domain discriminator head.

    The discriminator takes mean-pooled node embeddings per graph,
    passes through GRL -> MLP -> n_patients logits.
    """

    def __init__(self, base_model, n_patients, hidden_dim=128):
        super().__init__()
        self.base_model = base_model
        self.grl = GRL()
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_patients),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None, lambda_=0.0):
        """Forward pass returning boundary logits + domain logits.

        We intercept the base model forward to extract encoder embeddings
        before the boundary head is applied.
        """
        model = self.base_model

        # --- Replicate the encoder forward pass ---
        # Fold edge features into node features for non-native archs
        if model.arch != "mlp_only":
            x = model._project_edge_to_nodes(x, edge_index, edge_attr)

        # Prepare edge_attr for native-edge convs
        ea = None
        from model import _NATIVE_EDGE_ARCHS
        if model.arch in _NATIVE_EDGE_ARCHS and model.edge_dim > 0:
            ea = (
                edge_attr
                if edge_attr is not None
                else x.new_zeros(edge_index.size(1), model.edge_dim)
            )

        # Conv / MLP blocks
        for conv, norm in zip(model.convs, model.norms):
            x_in = x
            if model.arch == "mlp_only":
                x = conv(x)
            elif model.arch in _NATIVE_EDGE_ARCHS:
                x = conv(x, edge_index, ea)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=model.dropout, training=self.training)
            if model.use_residual and x_in.shape == x.shape:
                x = x + x_in

        # x is now the encoder output [N, hidden_dim]
        encoder_out = x

        # --- Boundary head (from base model) ---
        boundary_logits = model.boundary_head(encoder_out).squeeze(-1)

        # --- Domain head (via GRL) ---
        # Mean-pool node embeddings per graph using scatter
        if batch is not None:
            # Manual scatter_mean: sum then divide by count
            n_graphs = int(batch.max().item()) + 1
            pooled = encoder_out.new_zeros(n_graphs, self.hidden_dim)
            counts = encoder_out.new_zeros(n_graphs, 1)
            ones = encoder_out.new_ones(encoder_out.size(0), 1)

            idx_expand = batch.unsqueeze(1).expand_as(encoder_out)
            pooled.scatter_add_(0, idx_expand, encoder_out)
            counts.scatter_add_(0, batch.unsqueeze(1), ones)
            pooled = pooled / counts.clamp(min=1)
        else:
            pooled = encoder_out.mean(dim=0, keepdim=True)

        reversed_pooled = self.grl(pooled, lambda_)
        domain_logits = self.domain_head(reversed_pooled)

        return {
            "boundary_logits": boundary_logits,
            "domain_logits": domain_logits,
        }


# ===========================================================================
# Patient-ID assignment utilities
# ===========================================================================

def assign_patient_ids_from_dir(data_list, data_dir):
    """Assign integer patient_idx to each Data object based on filenames.

    The filenames follow the pattern bidmcXX_wYYYY.pt. We extract the patient
    prefix (everything before _w) and assign a stable integer ID.

    Returns:
        data_list (modified in place with .patient_idx attribute)
        pid_to_idx: dict mapping patient string ID to integer index
        n_patients: total number of unique patients
    """
    files = sorted(Path(data_dir).glob("*.pt"))
    pids = []
    for fpath in files:
        stem = fpath.stem
        if "_w" in stem:
            pid = stem[: stem.index("_w")]
        else:
            pid = stem.rsplit("_", 1)[0] if "_" in stem else stem
        pids.append(pid)

    unique_pids = sorted(set(pids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    assert len(files) == len(data_list), (
        f"Mismatch: {len(files)} files vs {len(data_list)} loaded Data objects"
    )

    for data_obj, pid in zip(data_list, pids):
        data_obj.patient_idx = torch.tensor(pid_to_idx[pid], dtype=torch.long)

    return data_list, pid_to_idx, len(unique_pids)


# ===========================================================================
# Custom training loop with domain adversarial loss
# ===========================================================================

def get_lambda_schedule(epoch, max_epoch, max_lambda, warmup_frac=0.5):
    """Linear ramp from 0 to max_lambda over the first warmup_frac of training."""
    if max_lambda <= 0:
        return 0.0
    warmup_epochs = int(max_epoch * warmup_frac)
    if warmup_epochs <= 0:
        return max_lambda
    progress = min(epoch / warmup_epochs, 1.0)
    return max_lambda * progress


def train_one_epoch_adversarial(
    model, loader, optimizer, config, device, lambda_grl
):
    """Train for one epoch with combined boundary + domain adversarial loss.

    Returns (mean_loss, mean_boundary_loss, mean_domain_loss, device_used).
    """
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dom = 0.0
    n_graphs = 0

    pw_val = config.get("pos_weight", 7.0)
    pos_weight = torch.tensor(pw_val, device=device)
    edge_drop_rate = config.get("edge_drop_rate", 0.0)

    for batch in loader:
        try:
            batch = batch.to(device)
            optimizer.zero_grad()

            # DropEdge
            ei = batch.edge_index
            ea = batch.edge_attr if hasattr(batch, "edge_attr") else None
            if edge_drop_rate > 0:
                ei, ea = drop_edges(ei, ea, edge_drop_rate)

            out = model(
                batch.x,
                ei,
                edge_attr=ea,
                batch=batch.batch if hasattr(batch, "batch") else None,
                lambda_=lambda_grl,
            )

            # Boundary loss (BCE with pos_weight)
            bce_loss = F.binary_cross_entropy_with_logits(
                out["boundary_logits"], batch.y, pos_weight=pos_weight
            )

            # Domain loss (cross-entropy on per-graph patient classification)
            if lambda_grl > 0 and "domain_logits" in out:
                # batch.patient_idx is already per-graph [B] from PyG batching
                # (scalar Data attributes are stacked into a 1D tensor)
                graph_patient = batch.patient_idx

                domain_loss = F.cross_entropy(
                    out["domain_logits"], graph_patient.long()
                )
            else:
                domain_loss = torch.tensor(0.0, device=device)

            # Combined loss: boundary BCE + lambda * domain adversarial
            # The GRL already negates the domain gradient w.r.t. encoder,
            # so we ADD the domain loss (gradient reversal happens inside)
            loss = bce_loss + lambda_grl * domain_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_bce += bce_loss.item() * batch.num_graphs
            total_dom += domain_loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

        except RuntimeError as e:
            err_msg = str(e).lower()
            if any(
                k in err_msg
                for k in (
                    "hip",
                    "cuda",
                    "device-side assert",
                    "invalid device function",
                    "out of memory",
                )
            ):
                print(f"  [GPU ERROR] {e}")
                if device != "cpu":
                    print("  Falling back to CPU for remainder of epoch")
                    device = "cpu"
                    model.cpu()
                    pos_weight = pos_weight.cpu()
                    for pg in optimizer.param_groups:
                        for p in pg["params"]:
                            state = optimizer.state.get(p, {})
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.cpu()
                    # Retry batch on CPU
                    batch = batch.to("cpu")
                    optimizer.zero_grad()
                    out = model(
                        batch.x,
                        batch.edge_index,
                        edge_attr=batch.edge_attr
                        if hasattr(batch, "edge_attr")
                        else None,
                        batch=batch.batch if hasattr(batch, "batch") else None,
                        lambda_=lambda_grl,
                    )
                    bce_loss = F.binary_cross_entropy_with_logits(
                        out["boundary_logits"], batch.y, pos_weight=pos_weight
                    )
                    loss = bce_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * batch.num_graphs
                    total_bce += bce_loss.item() * batch.num_graphs
                    n_graphs += batch.num_graphs
                else:
                    raise
            else:
                raise

    denom = max(n_graphs, 1)
    return total_loss / denom, total_bce / denom, total_dom / denom, device


def train_adversarial(
    config,
    train_dir,
    val_dir,
    n_patients,
    max_lambda,
    device="cuda",
    verbose=True,
):
    """Full adversarial training loop.

    Returns dict with best_val_f1, checkpoint path, training stats.
    """
    # Seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load data
    train_data = load_cached_graphs(train_dir)
    val_data = load_cached_graphs(val_dir)

    # Assign patient IDs
    train_data, train_pid_map, n_train_patients = assign_patient_ids_from_dir(
        train_data, train_dir
    )
    val_data, _, _ = assign_patient_ids_from_dir(val_data, val_dir)

    # Use number of training patients for domain head
    actual_n_patients = max(n_train_patients, n_patients)

    if verbose:
        print(
            f"  Train: {len(train_data)} graphs, {n_train_patients} patients; "
            f"Val: {len(val_data)} graphs"
        )

    # Build base model
    in_dim, edge_dim = compute_feature_dims(config)
    model_config = dict(config)
    model_config["in_dim"] = in_dim
    model_config["edge_dim"] = edge_dim
    base_model = build_model(model_config)
    hidden_dim = model_config.get("hidden_dim", 128)

    # Wrap with domain adversarial head
    model = DomainAdversarialModel(
        base_model, n_patients=actual_n_patients, hidden_dim=hidden_dim
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model params: {n_params} (includes domain head)")

    # Data loaders
    batch_size = config.get("batch_size", 32)
    train_loader = PyGLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = PyGLoader(val_data, batch_size=batch_size)

    # Optimizer
    lr = config.get("lr", 1e-3)
    wd = config.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    n_epochs = config.get("max_epochs", 100)
    patience = config.get("patience", 15)
    tol = config.get("tol_samples_train", 75)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    best_val_f1 = 0.0
    best_model_state = None
    best_epoch = 0
    no_improve = 0
    history = []
    final_epoch = 0

    for epoch in range(n_epochs):
        final_epoch = epoch
        t0 = time.time()

        # Lambda schedule: linear ramp over first 50% of training
        lambda_grl = get_lambda_schedule(epoch, n_epochs, max_lambda)

        train_loss, bce_loss, dom_loss, device = train_one_epoch_adversarial(
            model, train_loader, optimizer, config, device, lambda_grl
        )

        # Evaluate: create thin wrapper so standard evaluate() works
        model.eval()

        class _EvalWrapper(nn.Module):
            def __init__(self_, adv_model):
                super().__init__()
                self_.adv_model = adv_model

            def forward(self_, x, edge_index, edge_attr=None, batch=None):
                out = self_.adv_model(
                    x, edge_index, edge_attr=edge_attr, batch=batch, lambda_=0.0
                )
                return {"boundary_logits": out["boundary_logits"]}

        eval_model = _EvalWrapper(model)
        eval_model.eval()
        val_metrics = evaluate(eval_model, val_loader, device, tol_samples=tol)
        val_f1 = val_metrics["boundary_f1_600ms"]

        dt = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "bce_loss": bce_loss,
                "dom_loss": dom_loss,
                "lambda": lambda_grl,
                "val_f1": val_f1,
                "lr": optimizer.param_groups[0]["lr"],
                "dt": dt,
            }
        )

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                f"bce={bce_loss:.4f} dom={dom_loss:.4f} lam={lambda_grl:.3f} "
                f"val_f1={val_f1:.4f} lr={lr_now:.2e} "
                f"({dt:.1f}s)"
            )

        # Track best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            # Save only the base model state (for clean evaluation later)
            best_model_state = {
                k: v.cpu().clone()
                for k, v in model.base_model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step()

        if no_improve >= patience:
            if verbose:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improve for {patience} epochs)"
                )
            break

    # Save checkpoint
    ckpt_dir = os.path.join(os.path.dirname(train_dir), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if best_model_state is not None:
        torch.save(best_model_state, ckpt_path)
    else:
        torch.save(model.base_model.state_dict(), ckpt_path)

    if verbose:
        print(
            f"  Best val_f1={best_val_f1:.4f} at epoch {best_epoch}, "
            f"saved to {ckpt_path}"
        )

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "ckpt_path": ckpt_path,
        "n_epochs_run": final_epoch + 1,
        "history": history,
    }


# ===========================================================================
# Helpers
# ===========================================================================

def load_patients():
    from data_loader import load_all_patients
    return load_all_patients()


def load_splits():
    with open("results/splits.json") as f:
        return json.load(f)


# ===========================================================================
# Trial runner
# ===========================================================================

def run_trial(config_name, max_lambda, feat_config, train_config, seed,
              patients, splits):
    """Run a single adversarial training trial and evaluate on test."""
    base_dir = f"data/graphs/p95_{config_name}"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Cache graphs (skip if already done)
    if len(os.listdir(train_dir)) == 0:
        cache_split_graphs(patients, splits["train"], train_dir, config=feat_config)
    if len(os.listdir(val_dir)) == 0:
        cache_split_graphs(patients, splits["val"], val_dir, config=feat_config)
    if len(os.listdir(test_dir)) == 0:
        cache_split_graphs(patients, splits["test"], test_dir, config=feat_config)

    # Count patients in training set
    train_files = sorted(Path(train_dir).glob("*.pt"))
    train_pids = set()
    for fpath in train_files:
        stem = fpath.stem
        if "_w" in stem:
            train_pids.add(stem[: stem.index("_w")])
        else:
            train_pids.add(stem.rsplit("_", 1)[0] if "_" in stem else stem)
    n_patients = len(train_pids)

    in_dim, edge_dim = compute_feature_dims(feat_config)
    print(f"  Features: in_dim={in_dim}, edge_dim={edge_dim}")
    print(f"  max_lambda={max_lambda}, n_patients={n_patients}")

    full_config = dict(train_config)
    full_config.update(feat_config)
    full_config["in_dim"] = in_dim
    full_config["edge_dim"] = edge_dim
    full_config["seed"] = seed

    # Train
    result = train_adversarial(
        config=full_config,
        train_dir=train_dir,
        val_dir=val_dir,
        n_patients=n_patients,
        max_lambda=max_lambda,
        device="cuda",
    )

    # --- Test evaluation ---
    ckpt_path = result["ckpt_path"]
    model_config = dict(full_config)
    model_config["in_dim"] = in_dim
    model_config["edge_dim"] = edge_dim
    model = build_model(model_config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    eval_device = "cpu"

    val_data = load_cached_graphs(val_dir)
    test_data = load_cached_graphs(test_dir)
    val_loader = PyGLoader(val_data, batch_size=32)
    test_loader = PyGLoader(test_data, batch_size=32)

    # Val-optimized PP
    pp = optimize_threshold(model, val_loader, eval_device, tol_samples=75)
    val_opt_thresh, val_opt_nms = pp[0], pp[1]

    val_r = evaluate(
        model, val_loader, eval_device, tol_samples=75,
        threshold=val_opt_thresh, nms_dist=val_opt_nms,
    )
    test_r = evaluate(
        model, test_loader, eval_device, tol_samples=75,
        threshold=val_opt_thresh, nms_dist=val_opt_nms,
    )
    test_robust = evaluate(
        model, test_loader, eval_device, tol_samples=75,
        threshold=ROBUST_PP["threshold"], nms_dist=ROBUST_PP["nms_dist"],
    )
    test_raw = evaluate(
        model, test_loader, eval_device, tol_samples=75,
        threshold=0.5, nms_dist=0,
    )

    return {
        "config": config_name,
        "seed": seed,
        "max_lambda": max_lambda,
        "val_pp": val_r["boundary_f1_600ms"],
        "test_valpp": test_r["boundary_f1_600ms"],
        "test_robust": test_robust["boundary_f1_600ms"],
        "test_raw": test_raw["boundary_f1_600ms"],
        "best_val_f1": result["best_val_f1"],
        "best_epoch": result["best_epoch"],
        "n_epochs": result["n_epochs_run"],
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("P95: Patient-Aware Domain Adversarial Training")
    print("=" * 80)

    patients = load_patients()
    print(f"Loaded {len(patients)} patients")
    splits = load_splits()
    print(
        f"Splits: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    # Feature config: standard 6 features (baseline)
    feat_config = dict(DEFAULT_FEATURE_CONFIG)
    feat_config["multi_scale_cutoffs"] = None

    # Training config: TransformerConv 5L/128h/4H (matches campaign default)
    base_train = {
        "arch": "transformer",
        "hidden_dim": 128,
        "n_heads": 4,
        "n_layers": 5,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 100,
        "patience": 15,
        "pos_weight": 7.0,
        "batch_size": 32,
        "add_self_loops": False,
    }

    # Configs: (name, max_lambda)
    configs = [
        ("baseline", 0.0),
        ("grl_001", 0.01),
        ("grl_005", 0.05),
        ("grl_01", 0.1),
        ("grl_05", 0.5),
    ]

    all_results = []

    for config_name, max_lambda in configs:
        for seed in SEEDS:
            trial_name = f"{config_name}_s{seed}"
            print(f"\n{'=' * 60}")
            print(f"TRIAL: {trial_name}")
            print(f"{'=' * 60}")
            try:
                r = run_trial(
                    config_name,
                    max_lambda,
                    feat_config,
                    base_train,
                    seed,
                    patients,
                    splits,
                )
                all_results.append(r)
                print(
                    f"  RESULT: val_pp={r['val_pp']:.4f} "
                    f"test_valpp={r['test_valpp']:.4f} "
                    f"test_robust={r['test_robust']:.4f} "
                    f"test_raw={r['test_raw']:.4f}"
                )
            except Exception as e:
                print(f"ERROR in {trial_name}: {e}")
                traceback.print_exc()
                all_results.append(
                    {
                        "config": config_name,
                        "seed": seed,
                        "max_lambda": max_lambda,
                        "error": str(e),
                    }
                )

    # Summary
    print(f"\n{'=' * 80}")
    print("P95 SUMMARY")
    print(f"{'=' * 80}")
    for cname, _ in configs:
        cr = [r for r in all_results if r.get("config") == cname and "test_robust" in r]
        if cr:
            robust = [r["test_robust"] for r in cr]
            raw = [r["test_raw"] for r in cr]
            valpp = [r["test_valpp"] for r in cr]
            robust_mean = np.mean(robust)
            robust_std = np.std(robust)
            raw_mean = np.mean(raw)
            raw_std = np.std(raw)
            valpp_mean = np.mean(valpp)
            valpp_std = np.std(valpp)
            print(
                f"  {cname:12s}: "
                f"test_robust={robust_mean:.4f}+/-{robust_std:.4f}  "
                f"test_raw={raw_mean:.4f}+/-{raw_std:.4f}  "
                f"test_valpp={valpp_mean:.4f}+/-{valpp_std:.4f}"
            )

    # Save results
    os.makedirs("results/aorus", exist_ok=True)
    with open("results/aorus/p95_domain_adversarial.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/aorus/p95_domain_adversarial.json")
    print("DONE")


if __name__ == "__main__":
    main()
