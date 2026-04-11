#!/usr/bin/env python3
"""P107: Combined Multi-Scale DG + Soft-F1 Loss (P92+P99 synergy).

Combines:
  - P99 multi-scale coarse DG features (0.5+1.0Hz cutoffs → 14 node features)
  - P92 soft-F1 auxiliary loss (alpha=0.5)

Configs:
  baseline        — standard 6 features, BCE loss (control)
  ms_only         — 14 features (0.5+1.0Hz), BCE loss (P99 best)
  sf1_only        — 6 features, BCE+soft-F1(α=0.5) (P92 best)
  combo           — 14 features + BCE+soft-F1(α=0.5) (THE HYPOTHESIS)
  combo_a03       — 14 features + BCE+soft-F1(α=0.3)

Each config × 3 seeds = 15 total runs.
"""

import json
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from graph_features import cache_split_graphs, compute_feature_dims, DEFAULT_FEATURE_CONFIG
from train import train, load_cached_graphs, evaluate, optimize_threshold, build_model
from torch_geometric.loader import DataLoader as PyGLoader

SEEDS = [42, 123, 456]
ROBUST_PP = {"threshold": 0.15, "nms_dist": 250}


def load_patients():
    from data_loader import load_all_patients
    return load_all_patients()


def load_splits():
    with open("results/splits.json") as f:
        return json.load(f)


def run_trial(config_name, feat_config, train_config, seed, patients, splits):
    """Run a single training trial and evaluate on test."""
    base_dir = f"data/graphs/p107_{config_name}"
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

    in_dim, edge_dim = compute_feature_dims(feat_config)
    print(f"Features: in_dim={in_dim}, edge_dim={edge_dim}")
    if feat_config.get("multi_scale_cutoffs"):
        print(f"Multi-scale cutoffs: {feat_config['multi_scale_cutoffs']}")
    print(f"Loss: {train_config.get('loss_fn', 'bce')}, soft_f1_alpha={train_config.get('soft_f1_alpha', 0)}")

    full_config = {
        **train_config,
        **feat_config,
        "in_dim": in_dim,
        "edge_dim": edge_dim,
        "seed": seed,
    }

    result = train(
        config=full_config,
        train_dir=train_dir,
        val_dir=val_dir,
        device="cuda",
    )

    # --- Test evaluation ---
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_model.pt")
    model_config = {**full_config, "in_dim": in_dim, "edge_dim": edge_dim}
    model = build_model(model_config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    device = "cpu"

    val_data = load_cached_graphs(val_dir)
    test_data = load_cached_graphs(test_dir)
    val_loader = PyGLoader(val_data, batch_size=32)
    test_loader = PyGLoader(test_data, batch_size=32)

    # Val-optimized PP
    pp = optimize_threshold(model, val_loader, device, tol_samples=75)
    val_opt_thresh, val_opt_nms = pp[0], pp[1]

    val_r = evaluate(model, val_loader, device, tol_samples=75,
                     threshold=val_opt_thresh, nms_dist=val_opt_nms)
    test_r = evaluate(model, test_loader, device, tol_samples=75,
                      threshold=val_opt_thresh, nms_dist=val_opt_nms)
    test_robust = evaluate(model, test_loader, device, tol_samples=75,
                           threshold=ROBUST_PP["threshold"],
                           nms_dist=ROBUST_PP["nms_dist"])
    test_raw = evaluate(model, test_loader, device, tol_samples=75,
                        threshold=0.5, nms_dist=0)

    return {
        "config": config_name,
        "seed": seed,
        "val_pp": val_r["boundary_f1_600ms"],
        "test_valpp": test_r["boundary_f1_600ms"],
        "test_robust": test_robust["boundary_f1_600ms"],
        "test_raw": test_raw["boundary_f1_600ms"],
        "n_epochs": result["n_epochs_run"],
        "opt_thresh": result.get("opt_threshold"),
        "opt_nms": result.get("opt_nms_dist"),
    }


def main():
    print("P107: Combined Multi-Scale DG + Soft-F1 Loss")
    patients = load_patients()
    print(f"Loaded {len(patients)} patients")
    splits = load_splits()

    # Feature configs
    feat_base = dict(DEFAULT_FEATURE_CONFIG)
    feat_base["multi_scale_cutoffs"] = None

    feat_ms = dict(DEFAULT_FEATURE_CONFIG)
    feat_ms["multi_scale_cutoffs"] = [0.5, 1.0]

    # Training configs
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
        "loss_fn": "bce",
        "add_self_loops": False,
    }

    sf1_train = {**base_train, "loss_fn": "bce+soft_f1", "soft_f1_alpha": 0.5}
    sf1_03_train = {**base_train, "loss_fn": "bce+soft_f1", "soft_f1_alpha": 0.3}

    configs = [
        ("baseline",   feat_base, base_train),
        ("ms_only",    feat_ms,   base_train),
        ("sf1_only",   feat_base, sf1_train),
        ("combo",      feat_ms,   sf1_train),
        ("combo_a03",  feat_ms,   sf1_03_train),
    ]

    all_results = []
    for config_name, feat_config, train_config in configs:
        for seed in SEEDS:
            trial_name = f"{config_name}_s{seed}"
            print(f"\n{'='*60}")
            print(f"TRIAL: {trial_name}")
            print(f"{'='*60}")
            try:
                r = run_trial(config_name, feat_config, train_config, seed, patients, splits)
                all_results.append(r)
                print(f"  RESULT: val_pp={r['val_pp']:.4f} test_valpp={r['test_valpp']:.4f} "
                      f"test_robust={r['test_robust']:.4f} test_raw={r['test_raw']:.4f}")
            except Exception as e:
                print(f"ERROR in {trial_name}: {e}")
                traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print("P107 SUMMARY")
    print(f"{'='*80}")
    for cname in ["baseline", "ms_only", "sf1_only", "combo", "combo_a03"]:
        cr = [r for r in all_results if r["config"] == cname]
        if cr:
            robust = [r["test_robust"] for r in cr]
            raw = [r["test_raw"] for r in cr]
            print(f"  {cname:20s}: test_robust={np.mean(robust):.4f}±{np.std(robust):.4f} "
                  f"test_raw={np.mean(raw):.4f}±{np.std(raw):.4f}")

    os.makedirs("results/aorus", exist_ok=True)
    with open("results/aorus/p107_combo.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/aorus/p107_combo.json")
    print("DONE")


if __name__ == "__main__":
    main()
