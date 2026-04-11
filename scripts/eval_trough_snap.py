"""Evaluate trough snapping post-processing refinement (P64).

Usage:
    python scripts/eval_trough_snap.py [--model-path PATH] [--data-dir DIR]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from model import build_model
from graph_features import cache_split_graphs
from train import (
    _collect_predictions, compute_boundary_f1, compute_rate_mae,
    _apply_post_processing, trough_snap_1d, optimize_threshold
)


def evaluate_with_snap(graph_preds, threshold, nms_dist, top_n,
                       snap_window, snap_mode, tol_samples=75, fs=125):
    """Evaluate with trough snapping on pre-collected predictions."""
    all_f1s = []
    all_rate_maes = []
    for entry in graph_preds:
        g_scores, g_bars, g_labels = entry[0], entry[1], entry[2]
        g_feats = entry[3] if len(entry) > 3 else None
        
        pred_troughs = _apply_post_processing(
            g_scores, g_bars, threshold, nms_dist, top_n,
            g_features=g_feats,
            snap_window=snap_window, snap_mode=snap_mode)
        
        gt_troughs = g_bars[g_labels > 0.5]
        all_f1s.append(compute_boundary_f1(pred_troughs, gt_troughs, tol_samples))
        mae = compute_rate_mae(pred_troughs, gt_troughs, fs)
        if mae != float('inf'):
            all_rate_maes.append(mae)
    
    return {
        'f1': float(np.mean(all_f1s)) if all_f1s else 0.0,
        'rate_mae': float(np.mean(all_rate_maes)) if all_rate_maes else float('inf'),
        'n_graphs': len(graph_preds),
    }


def sweep_snap_params(graph_preds, base_threshold, base_nms, base_top_n, tol_samples=75):
    """Grid search over snap parameters."""
    snap_windows = [0, 15, 25, 35, 50, 75]
    snap_modes = ['nearest_trough', 'highest_score_trough', 'deepest_trough']
    
    results = []
    for sw in snap_windows:
        for sm in snap_modes:
            if sw == 0 and sm != 'nearest_trough':
                continue  # skip redundant no-snap entries
            r = evaluate_with_snap(
                graph_preds, base_threshold, base_nms, base_top_n,
                sw, sm, tol_samples)
            results.append({
                'snap_window': sw, 'snap_mode': sm,
                'f1': r['f1'], 'rate_mae': r['rate_mae']
            })
            print(f"  snap_window={sw:3d} mode={sm:25s} → F1={r['f1']:.4f} rate_MAE={r['rate_mae']:.3f}")
    
    return sorted(results, key=lambda x: -x['f1'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='data/graphs/checkpoints/P64_model_s789.pt')
    parser.add_argument('--data-dir', default='data/graphs')
    parser.add_argument('--threshold', type=float, default=0.02)
    parser.add_argument('--nms-dist', type=int, default=175)
    parser.add_argument('--top-n', type=int, default=6)
    args = parser.parse_args()
    
    device = torch.device('cpu')  # Safe for evaluation
    
    # Load model
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = build_model(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load val graphs
    for split_name in ['val', 'val_adversarial']:
        split_dir = Path(args.data_dir) / split_name
        if not split_dir.exists():
            print(f"Skipping {split_name}: {split_dir} not found")
            continue
        
        graphs = []
        for pt_file in sorted(split_dir.glob('*.pt')):
            graphs.append(torch.load(pt_file, weights_only=False))
        
        if not graphs:
            print(f"No graphs found in {split_dir}")
            continue
        
        loader = DataLoader(graphs, batch_size=32, shuffle=False)
        
        # Collect predictions with features
        preds = _collect_predictions(model, loader, device, include_features=True)
        
        print(f"\n=== {split_name} ({len(preds)} graphs) ===")
        print(f"Baseline PP: threshold={args.threshold}, NMS={args.nms_dist}, top_n={args.top_n}")
        
        baseline = evaluate_with_snap(preds, args.threshold, args.nms_dist, args.top_n,
                                      snap_window=0, snap_mode='nearest_trough')
        print(f"Baseline F1: {baseline['f1']:.4f}, Rate MAE: {baseline['rate_mae']:.3f}")
        
        print("\nSnap parameter sweep:")
        ranked = sweep_snap_params(preds, args.threshold, args.nms_dist, args.top_n)
        
        if ranked:
            best = ranked[0]
            print(f"\nBest: snap_window={best['snap_window']}, mode={best['snap_mode']}")
            print(f"  F1={best['f1']:.4f} (Δ={best['f1']-baseline['f1']:+.4f})")
            print(f"  Rate MAE={best['rate_mae']:.3f}")


if __name__ == '__main__':
    main()
