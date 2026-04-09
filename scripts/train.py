"""Training loop for DG-GNN respiratory boundary detection.

Reads a config dict specifying the full experiment configuration:
graph construction, features, architecture, training hyperparameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import build_model, count_parameters
from graph_features import compute_feature_dims, cache_split_graphs

def load_cached_graphs(data_dir: str) -> list:
    """Load all .pt files from a directory."""
    from torch_geometric.data import Data
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.pt"))
    return [torch.load(f, weights_only=False) for f in files]


def _rebuild_graphs_if_needed(
    data_dir: str, split_name: str, config: dict, verbose: bool = True
) -> str:
    """Rebuild cached graphs when feature dims don't match config.

    Returns the (possibly updated) data_dir path.
    """
    expected_in, expected_edge = compute_feature_dims(config)
    existing = sorted(Path(data_dir).glob("*.pt"))
    if existing:
        sample = torch.load(existing[0], weights_only=False)
        cached_in = sample.x.shape[1] if sample.x.dim() == 2 else 0
        cached_edge = sample.edge_attr.shape[1] if sample.edge_attr.dim() == 2 else 0
        if cached_in == expected_in and cached_edge == expected_edge:
            return data_dir  # dims match, use as-is
    else:
        cached_in, cached_edge = 0, 0

    if verbose:
        print(f"  Rebuilding {split_name} graphs: cached=({cached_in},{cached_edge}), "
              f"need=({expected_in},{expected_edge})")

    from data_loader import load_all_patients, get_splits, profile_patient
    patients = load_all_patients(verbose=False)
    splits = get_splits()

    # Determine which patient IDs to use
    if split_name == "val_adversarial":
        split_ids = [
            pid for pid in splits["val"]
            if profile_patient(patients[pid]).get("is_adversarial", False)
        ]
    elif split_name in splits:
        split_ids = splits[split_name]
    else:
        # Infer from existing filenames
        split_ids = sorted(set(
            f.stem.rsplit("_w", 1)[0] for f in Path(data_dir).glob("*.pt")
        ))

    wb = config.get("window_breaths", 6)
    # Clear old files
    for f in Path(data_dir).glob("*.pt"):
        f.unlink()
    cache_split_graphs(patients, split_ids, data_dir, config=config,
                       window_breaths=wb)
    if verbose:
        n_new = len(list(Path(data_dir).glob("*.pt")))
        print(f"  Rebuilt {n_new} graphs for {split_name}")
    return data_dir

def compute_boundary_f1(pred_troughs, gt_troughs, tol_samples=75):
    """Compute boundary F1 at ±tol_samples tolerance.
    
    Uses greedy closest-first matching (same as common.py boundary_f1).
    """
    if len(pred_troughs) == 0 or len(gt_troughs) == 0:
        return 0.0
    
    pairs = []
    for i, gt in enumerate(gt_troughs):
        for j, pred in enumerate(pred_troughs):
            dist = abs(float(pred) - float(gt))
            if dist <= tol_samples:
                pairs.append((dist, i, j))
    pairs.sort()
    used_gt, used_pred = set(), set()
    matched = 0
    for _, i, j in pairs:
        if i not in used_gt and j not in used_pred:
            matched += 1
            used_gt.add(i)
            used_pred.add(j)
    
    precision = matched / len(pred_troughs) if len(pred_troughs) > 0 else 0
    recall = matched / len(gt_troughs) if len(gt_troughs) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_rate_mae(pred_troughs, gt_troughs, fs=125):
    """Compute respiratory rate MAE in bpm from trough positions."""
    if len(pred_troughs) < 2 or len(gt_troughs) < 2:
        return float('inf')
    pred_intervals = np.diff(sorted(pred_troughs)) / fs
    gt_intervals = np.diff(sorted(gt_troughs)) / fs
    pred_rate = 60.0 / np.mean(pred_intervals) if np.mean(pred_intervals) > 0 else 0
    gt_rate = 60.0 / np.mean(gt_intervals) if np.mean(gt_intervals) > 0 else 0
    return abs(pred_rate - gt_rate)

def train_one_epoch(model, loader, optimizer, config, device):
    """Train for one epoch. Returns (mean_loss, device_used).
    
    Robust to GPU crashes: if a HIP/CUDA error occurs mid-batch,
    moves model+data to CPU and retries remaining batches.
    """
    model.train()
    total_loss = 0
    n_graphs = 0
    
    pos_weight = torch.tensor(config.get('pos_weight', 7.0), device=device)
    rate_w = config.get('loss_rate_weight', 0.0)
    type_w = config.get('loss_type_weight', 0.0)
    
    for batch in loader:
        try:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, 
                         edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                         batch=batch.batch if hasattr(batch, 'batch') else None)
            
            # Boundary loss (main)
            bce = F.binary_cross_entropy_with_logits(
                out['boundary_logits'], batch.y, pos_weight=pos_weight)
            
            loss = bce
            
            # Rate aux loss
            if rate_w > 0 and 'rate_pred' in out and hasattr(batch, 'rate_target'):
                rate_loss = F.mse_loss(out['rate_pred'].squeeze(), batch.rate_target)
                loss = loss + rate_w * rate_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs
        except RuntimeError as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ('hip', 'cuda', 'device-side assert', 'invalid device function', 'out of memory')):
                print(f"  [GPU ERROR] {e}")
                if device != 'cpu':
                    print("  Falling back to CPU for remainder of epoch")
                    device = 'cpu'
                    model.cpu()
                    pos_weight = pos_weight.cpu()
                    for pg in optimizer.param_groups:
                        for p in pg['params']:
                            state = optimizer.state.get(p, {})
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.cpu()
                    # Retry this batch on CPU
                    batch = batch.to('cpu')
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index,
                                 edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                                 batch=batch.batch if hasattr(batch, 'batch') else None)
                    bce = F.binary_cross_entropy_with_logits(
                        out['boundary_logits'], batch.y, pos_weight=pos_weight)
                    loss = bce
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * batch.num_graphs
                    n_graphs += batch.num_graphs
                else:
                    raise
            else:
                raise
    
    return total_loss / max(n_graphs, 1), device

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, tol_samples=75, fs=125):
    """Evaluate model on a data loader. Returns dict with metrics."""
    model.eval()
    all_f1s = []
    all_rate_maes = []
    total_loss = 0
    n_graphs = 0
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index,
                     edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                     batch=batch.batch if hasattr(batch, 'batch') else None)
        
        # Loss
        bce = F.binary_cross_entropy_with_logits(out['boundary_logits'], batch.y)
        total_loss += bce.item() * batch.num_graphs
        n_graphs += batch.num_graphs
        
        # Per-graph F1
        logits = out['boundary_logits'].cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        labels = batch.y.cpu().numpy()
        node_bars = batch.node_bars.cpu().numpy()
        batch_ids = batch.batch.cpu().numpy() if hasattr(batch, 'batch') and batch.batch is not None else np.zeros(len(logits), dtype=int)
        
        for g in range(batch.num_graphs):
            mask = batch_ids == g
            g_scores = scores[mask]
            g_bars = node_bars[mask]
            g_labels = labels[mask]
            
            pred_troughs = g_bars[g_scores > threshold]
            gt_troughs = g_bars[g_labels > 0.5]
            
            f1 = compute_boundary_f1(pred_troughs, gt_troughs, tol_samples)
            all_f1s.append(f1)
            
            mae = compute_rate_mae(pred_troughs, gt_troughs, fs)
            if mae != float('inf'):
                all_rate_maes.append(mae)
    
    return {
        'boundary_f1_600ms': float(np.mean(all_f1s)) if all_f1s else 0.0,
        'rate_mae_bpm': float(np.mean(all_rate_maes)) if all_rate_maes else float('inf'),
        'loss': total_loss / max(n_graphs, 1),
        'n_graphs': n_graphs,
    }

def make_scheduler(optimizer, config, n_epochs):
    """Create LR scheduler from config."""
    sched_type = config.get('scheduler', 'cosine')
    warmup = config.get('warmup_epochs', 0)
    
    if sched_type == 'none':
        return None
    elif sched_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif sched_type == 'cosine_warmup':
        # Linear warmup then cosine
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(n_epochs - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sched_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)
    return None

def train(config: dict, train_dir: str, val_dir: str,
          val_adv_dir: str | None = None,
          device: str = 'cuda', verbose: bool = True) -> dict:
    """Full training loop.
    
    Args:
        config: full experiment config dict
        train_dir: path to training .pt files
        val_dir: path to validation .pt files
        val_adv_dir: optional path to adversarial validation .pt files
        device: 'cuda' or 'cpu'
        verbose: print progress
    
    Returns dict with:
        best_val_f1, best_val_rate_mae, final_metrics, training_history
    """
    # Load data — rebuild graphs if feature dims don't match config
    train_dir = _rebuild_graphs_if_needed(train_dir, "train", config, verbose)
    val_dir = _rebuild_graphs_if_needed(val_dir, "val", config, verbose)
    if val_adv_dir:
        val_adv_dir = _rebuild_graphs_if_needed(val_adv_dir, "val_adversarial", config, verbose)

    train_data = load_cached_graphs(train_dir)
    val_data = load_cached_graphs(val_dir)
    val_adv_data = load_cached_graphs(val_adv_dir) if val_adv_dir else []
    
    if verbose:
        print(f"Train: {len(train_data)} graphs, Val: {len(val_data)}, Val-Adv: {len(val_adv_data)}")
    
    # Build model
    in_dim, edge_dim = compute_feature_dims(config)
    model_config = {**config, 'in_dim': in_dim, 'edge_dim': edge_dim}
    model = build_model(model_config).to(device)
    
    if verbose:
        print(f"Model: {config.get('arch', 'gat')}, params: {count_parameters(model)}")
    
    # Data loaders
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    val_adv_loader = DataLoader(val_adv_data, batch_size=batch_size) if val_adv_data else None
    
    # Optimizer
    opt_name = config.get('optimizer', 'adamw')
    lr = config.get('lr', 1e-3)
    wd = config.get('weight_decay', 1e-4)
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'sgd_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    n_epochs = config.get('max_epochs', 100)
    patience = config.get('patience', 15)
    scheduler = make_scheduler(optimizer, config, n_epochs)
    tol = config.get('tol_samples_train', 75)
    
    best_val_f1 = 0.0
    best_metrics = {}
    no_improve = 0
    history = []
    
    gpu_fell_back = False
    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss, device = train_one_epoch(model, train_loader, optimizer, config, device)
        if not gpu_fell_back and device == 'cpu' and config.get('_original_device', 'cuda') != 'cpu':
            gpu_fell_back = True
            print("  [WARN] Continuing training on CPU after GPU fallback")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device, tol_samples=tol)
        val_f1 = val_metrics['boundary_f1_600ms']
        
        # Adversarial eval
        if val_adv_loader:
            val_adv_metrics = evaluate(model, val_adv_loader, device, tol_samples=tol)
            # Weighted aggregate (55% val, 45% adv)
            agg_f1 = 0.55 * val_f1 + 0.45 * val_adv_metrics['boundary_f1_600ms']
        else:
            val_adv_metrics = {}
            agg_f1 = val_f1
        
        dt = time.time() - t0
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_f1': val_f1,
            'val_rate_mae': val_metrics['rate_mae_bpm'],
            'agg_f1': agg_f1,
            'lr': optimizer.param_groups[0]['lr'],
            'dt': dt,
        })
        
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} val_f1={val_f1:.4f} agg={agg_f1:.4f} lr={optimizer.param_groups[0]['lr']:.2e} ({dt:.1f}s)")
        
        # Track best
        if agg_f1 > best_val_f1:
            best_val_f1 = agg_f1
            best_metrics = {
                'boundary_f1_600ms': agg_f1,
                'val_f1': val_f1,
                'rate_mae_bpm': val_metrics['rate_mae_bpm'],
                'epoch': epoch,
            }
            if val_adv_metrics:
                best_metrics['val_adv_f1'] = val_adv_metrics['boundary_f1_600ms']
                best_metrics['val_adv_rate_mae'] = val_adv_metrics['rate_mae_bpm']
            no_improve = 0
        else:
            no_improve += 1
        
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(agg_f1)
            else:
                scheduler.step()
        
        # Early stopping
        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (no improve for {patience} epochs)")
            break
    
    return {
        'best_val_f1': best_val_f1,
        'best_metrics': best_metrics,
        'history': history,
        'n_epochs_run': len(history),
        'n_params': count_parameters(model),
    }
