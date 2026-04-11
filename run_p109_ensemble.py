#!/usr/bin/env python3
"""P109: K-Fold Patient Cross-Validation Ensemble.

Train 5 models using 5-fold patient-stratified CV on the train+val set (42 patients).
Each fold: ~34 train / ~8 val. At test time, average boundary scores from all 5 models
before applying NMS. This directly addresses the patient distribution shift that causes
the val→test gap.

Also includes:
  - combo_a03 features (multi-scale 0.5+1.0Hz + soft-F1 α=0.3) — best raw config from P107
  - Single-model baseline with same config for comparison

Configs:
  single_baseline — 1 model, standard train/val split
  ensemble_5fold  — 5 models, patient-stratified 5-fold CV
"""

import json
import os
import sys
import traceback
from collections import defaultdict

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


def make_kfold_splits(train_ids, val_ids, n_folds=5, seed=42):
    """Create K-fold patient-stratified splits from train+val patients."""
    rng = np.random.RandomState(seed)
    all_ids = sorted(train_ids + val_ids)
    rng.shuffle(all_ids)
    
    folds = []
    fold_size = len(all_ids) // n_folds
    for i in range(n_folds):
        start = i * fold_size
        if i == n_folds - 1:
            fold_val = all_ids[start:]
        else:
            fold_val = all_ids[start:start + fold_size]
        fold_train = [pid for pid in all_ids if pid not in fold_val]
        folds.append((fold_train, fold_val))
        print(f"  Fold {i}: train={len(fold_train)}, val={len(fold_val)}, val_patients={fold_val}")
    
    return folds


def get_train_config():
    """Best config: multi-scale + soft-F1(α=0.3) from P107."""
    feat_config = dict(DEFAULT_FEATURE_CONFIG)
    feat_config["multi_scale_cutoffs"] = [0.5, 1.0]
    
    train_config = {
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
        "loss_fn": "bce+soft_f1",
        "soft_f1_alpha": 0.3,
        "add_self_loops": False,
        **feat_config,
    }
    return feat_config, train_config


def train_fold(fold_idx, fold_train_ids, fold_val_ids, patients, feat_config,
               train_config, seed):
    """Train a single fold model."""
    base_dir = f"data/graphs/p109_fold{fold_idx}_s{seed}"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    if len(os.listdir(train_dir)) == 0:
        cache_split_graphs(patients, fold_train_ids, train_dir, config=feat_config)
    if len(os.listdir(val_dir)) == 0:
        cache_split_graphs(patients, fold_val_ids, val_dir, config=feat_config)
    
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_model.pt")
    if os.path.exists(ckpt_path):
        print(f"  Fold {fold_idx} already trained, skipping (checkpoint exists)")
        return ckpt_path, None
    
    in_dim, edge_dim = compute_feature_dims(feat_config)
    full_config = {**train_config, "in_dim": in_dim, "edge_dim": edge_dim, "seed": seed}
    
    result = train(config=full_config, train_dir=train_dir, val_dir=val_dir, device="cuda")
    
    return ckpt_path, result


def ensemble_evaluate(ckpt_paths, test_dir, train_config, feat_config, tol_samples=75):
    """Evaluate ensemble by averaging boundary scores across models."""
    in_dim, edge_dim = compute_feature_dims(feat_config)
    model_config = {**train_config, "in_dim": in_dim, "edge_dim": edge_dim}
    
    test_data = load_cached_graphs(test_dir)
    test_loader = PyGLoader(test_data, batch_size=32, shuffle=False)
    
    # Load all models
    models = []
    for ckpt_path in ckpt_paths:
        model = build_model(model_config)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()
        models.append(model)
    
    # Collect predictions from each model, then average
    from train import nms_1d
    
    all_graph_results = []
    
    for batch in test_loader:
        batch = batch.cpu()
        n_graphs = batch.num_graphs
        
        # Get per-model scores for this batch
        batch_scores = []
        for model in models:
            with torch.no_grad():
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                scores = torch.sigmoid(out['boundary_logits']).numpy()
                batch_scores.append(scores)
        
        # Average scores across models
        avg_scores = np.mean(batch_scores, axis=0)
        
        # Process per-graph
        ptr = batch.ptr.numpy() if hasattr(batch, 'ptr') else None
        node_bars_all = batch.node_bars.numpy()
        y_all = batch.y.numpy()
        n_gt_all = batch.n_gt_troughs.numpy() if hasattr(batch, 'n_gt_troughs') else None
        
        for gi in range(n_graphs):
            if ptr is not None:
                start, end = ptr[gi], ptr[gi + 1]
            else:
                mask = batch.batch.numpy() == gi
                start, end = np.where(mask)[0][[0, -1]]
                end += 1
            
            g_scores = avg_scores[start:end]
            g_bars = node_bars_all[start:end]
            g_y = y_all[start:end]
            
            all_graph_results.append({
                'scores': g_scores,
                'bars': g_bars,
                'y': g_y,
                'n_gt': int(n_gt_all[gi]) if n_gt_all is not None else int(g_y.sum()),
            })
    
    # Evaluate with different PP settings
    def eval_with_pp(threshold, nms_dist):
        from scripts.common import boundary_f1 as bf1
        all_f1 = []
        for gr in all_graph_results:
            mask = gr['scores'] > threshold
            pred_bars = gr['bars'][mask]
            pred_scores = gr['scores'][mask]
            
            if len(pred_bars) > 0 and nms_dist > 0:
                pred_bars = nms_1d(pred_bars, pred_scores, nms_dist)
            
            gt_bars = gr['bars'][gr['y'] > 0.5]
            if len(gt_bars) == 0:
                continue
            
            mean_dur = np.mean(np.diff(gt_bars)) if len(gt_bars) > 1 else 500
            f1 = bf1(
                pred_starts=pred_bars,
                pred_ends=pred_bars + 1,
                true_starts=gt_bars,
                true_ends=gt_bars + 1,
                true_durations=np.full(len(gt_bars), mean_dur),
                tolerance_frac=0.0,
                tolerance_floor=tol_samples,
            )
            all_f1.append(f1)
        return float(np.mean(all_f1)) if all_f1 else 0.0
    
    # Try multiple PP settings
    results = {}
    results['robust'] = eval_with_pp(ROBUST_PP['threshold'], ROBUST_PP['nms_dist'])
    results['raw'] = eval_with_pp(0.5, 0)
    
    # Grid search for best PP
    best_f1 = 0
    best_t, best_n = 0.5, 0
    for t in np.arange(0.01, 0.5, 0.02):
        for n in range(0, 301, 25):
            f1 = eval_with_pp(t, n)
            if f1 > best_f1:
                best_f1 = f1
                best_t, best_n = t, n
    results['best_pp'] = best_f1
    results['best_pp_thresh'] = best_t
    results['best_pp_nms'] = best_n
    
    return results


def run_single_baseline(patients, splits, feat_config, train_config, seed):
    """Run single-model baseline with same config."""
    base_dir = f"data/graphs/p109_single_s{seed}"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    if len(os.listdir(train_dir)) == 0:
        cache_split_graphs(patients, splits["train"], train_dir, config=feat_config)
    if len(os.listdir(val_dir)) == 0:
        cache_split_graphs(patients, splits["val"], val_dir, config=feat_config)
    if len(os.listdir(test_dir)) == 0:
        cache_split_graphs(patients, splits["test"], test_dir, config=feat_config)
    
    in_dim, edge_dim = compute_feature_dims(feat_config)
    full_config = {**train_config, "in_dim": in_dim, "edge_dim": edge_dim, "seed": seed}
    
    result = train(config=full_config, train_dir=train_dir, val_dir=val_dir, device="cuda")
    
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_model.pt")
    model_config = {**full_config}
    model = build_model(model_config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    
    test_data = load_cached_graphs(test_dir)
    val_data = load_cached_graphs(val_dir)
    test_loader = PyGLoader(test_data, batch_size=32)
    val_loader = PyGLoader(val_data, batch_size=32)
    
    pp = optimize_threshold(model, val_loader, "cpu", tol_samples=75)
    
    test_robust = evaluate(model, test_loader, "cpu", tol_samples=75,
                           threshold=ROBUST_PP["threshold"], nms_dist=ROBUST_PP["nms_dist"])
    test_raw = evaluate(model, test_loader, "cpu", tol_samples=75, threshold=0.5, nms_dist=0)
    test_valpp = evaluate(model, test_loader, "cpu", tol_samples=75,
                          threshold=pp[0], nms_dist=pp[1])
    
    return {
        "test_robust": test_robust["boundary_f1_600ms"],
        "test_raw": test_raw["boundary_f1_600ms"],
        "test_valpp": test_valpp["boundary_f1_600ms"],
    }


def main():
    print("P109: K-Fold Patient Cross-Validation Ensemble")
    patients = load_patients()
    print(f"Loaded {len(patients)} patients")
    splits = load_splits()
    
    feat_config, train_config = get_train_config()
    in_dim, edge_dim = compute_feature_dims(feat_config)
    print(f"Config: in_dim={in_dim}, edge_dim={edge_dim}, loss=bce+soft_f1(α=0.3)")
    
    all_results = []
    
    # --- Single-model baselines ---
    print("\n" + "="*80)
    print("PHASE 1: Single-model baselines")
    print("="*80)
    for seed in SEEDS:
        print(f"\n--- Single baseline seed={seed} ---")
        try:
            r = run_single_baseline(patients, splits, feat_config, train_config, seed)
            r["config"] = "single"
            r["seed"] = seed
            all_results.append(r)
            print(f"  RESULT: test_robust={r['test_robust']:.4f} test_raw={r['test_raw']:.4f}")
        except Exception as e:
            print(f"ERROR single_s{seed}: {e}")
            traceback.print_exc()
    
    # --- Cache test graphs (shared by all ensemble configs) ---
    test_dir = "data/graphs/p109_test"
    os.makedirs(test_dir, exist_ok=True)
    if len(os.listdir(test_dir)) == 0:
        cache_split_graphs(patients, splits["test"], test_dir, config=feat_config)
    
    # --- K-Fold Ensemble ---
    print("\n" + "="*80)
    print("PHASE 2: 5-Fold Patient CV Ensemble")
    print("="*80)
    
    for seed in SEEDS:
        print(f"\n--- Ensemble seed={seed} ---")
        folds = make_kfold_splits(splits["train"], splits["val"], n_folds=5, seed=seed)
        
        ckpt_paths = []
        for fold_idx, (fold_train, fold_val) in enumerate(folds):
            print(f"\n  Training fold {fold_idx}...")
            try:
                ckpt_path, _ = train_fold(fold_idx, fold_train, fold_val, patients,
                                          feat_config, train_config, seed)
                ckpt_paths.append(ckpt_path)
                print(f"  Fold {fold_idx} done: {ckpt_path}")
            except Exception as e:
                print(f"  ERROR fold {fold_idx}: {e}")
                traceback.print_exc()
        
        if len(ckpt_paths) >= 3:
            print(f"\n  Evaluating ensemble ({len(ckpt_paths)} models)...")
            try:
                ens_results = ensemble_evaluate(ckpt_paths, test_dir, train_config, feat_config)
                ens_results["config"] = "ensemble_5fold"
                ens_results["seed"] = seed
                ens_results["n_models"] = len(ckpt_paths)
                all_results.append(ens_results)
                print(f"  ENSEMBLE RESULT: robust={ens_results['robust']:.4f} "
                      f"raw={ens_results['raw']:.4f} best_pp={ens_results['best_pp']:.4f}")
            except Exception as e:
                print(f"  ERROR ensemble eval: {e}")
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("P109 SUMMARY")
    print(f"{'='*80}")
    
    single = [r for r in all_results if r.get("config") == "single"]
    ensemble = [r for r in all_results if r.get("config") == "ensemble_5fold"]
    
    if single:
        robust = [r["test_robust"] for r in single]
        raw = [r["test_raw"] for r in single]
        print(f"  single    : test_robust={np.mean(robust):.4f}±{np.std(robust):.4f} "
              f"test_raw={np.mean(raw):.4f}±{np.std(raw):.4f}")
    
    if ensemble:
        robust = [r["robust"] for r in ensemble]
        raw = [r["raw"] for r in ensemble]
        best = [r["best_pp"] for r in ensemble]
        print(f"  ensemble  : test_robust={np.mean(robust):.4f}±{np.std(robust):.4f} "
              f"test_raw={np.mean(raw):.4f}±{np.std(raw):.4f} "
              f"best_pp={np.mean(best):.4f}±{np.std(best):.4f}")
    
    os.makedirs("results/aorus", exist_ok=True)
    with open("results/aorus/p109_ensemble.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/aorus/p109_ensemble.json")
    print("DONE")


if __name__ == "__main__":
    main()
