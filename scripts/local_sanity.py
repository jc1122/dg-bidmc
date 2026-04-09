#!/usr/bin/env python3
"""Local sanity check for DG-GNN respiratory boundary detection.

Validates the full pipeline: load data → build graph → extract features →
forward pass → compute metrics. Must complete in < 60 seconds.

Usage:
    python3 scripts/local_sanity.py --fast
"""

import sys
import os
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

def run_sanity(fast: bool = True):
    """Run sanity checks. Returns True if all pass."""
    t0 = time.time()
    results = {}

    # 1. Load one patient (bidmc01) via wfdb
    print("[1/6] Loading bidmc01...")
    import numpy as np
    import wfdb
    from scipy.signal import butter, filtfilt

    rec = wfdb.rdrecord('bidmc01', pn_dir='bidmc')
    ann = wfdb.rdann('bidmc01', 'breath', pn_dir='bidmc')
    resp = rec.p_signal[:, 0]
    fs = rec.fs
    peaks = ann.sample[::2]

    b, a = butter(4, 2.0 / (fs/2), 'low')
    resp_lp = np.ascontiguousarray(filtfilt(b, a, resp - resp.mean()), dtype=np.float64)

    troughs = []
    for i in range(len(peaks) - 1):
        p1, p2 = int(peaks[i]), int(peaks[i+1])
        trough = p1 + int(np.argmin(resp_lp[p1:p2]))
        troughs.append(trough)

    print(f"  signal: {len(resp)} samples, {len(peaks)} peaks, {len(troughs)} troughs")
    results['load'] = True

    # 2. Build DG graph (2 variants)
    print("[2/6] Building DG graphs (standard + trough_emphasis)...")
    from graph_features import build_graph_variant, preprocess_signal_for_dg

    sig_lp = preprocess_signal_for_dg(resp)
    graph_std = build_graph_variant(sig_lp, variant="standard")
    graph_te = build_graph_variant(sig_lp, variant="trough_emphasis")

    n_nodes_std = len(graph_std.node_bar)
    n_nodes_te = len(graph_te.node_bar)
    print(f"  standard: {n_nodes_std} nodes, trough_emphasis: {n_nodes_te} nodes")
    assert n_nodes_std > 5, "too few nodes in standard graph"
    results['graph'] = True

    # 3. Extract features (all motifs metrics enabled)
    print("[3/6] Extracting features...")
    from graph_features import extract_graph_data, DEFAULT_FEATURE_CONFIG

    config = dict(DEFAULT_FEATURE_CONFIG)
    # Enable all features for sanity check
    for k in config:
        if k.startswith('feat_') or k.startswith('edge_feat_'):
            config[k] = True
    config['graph_variant'] = 'standard'

    # Use a window of the first ~6 breaths
    n_breaths = min(6, len(troughs))
    if n_breaths >= 2:
        window_end = troughs[n_breaths - 1] + 200
        window_end = min(window_end, len(resp))
        window_sig = resp[:window_end]
        window_troughs = [t for t in troughs if t < window_end]
    else:
        window_sig = resp[:5000]
        window_troughs = troughs[:3]

    data = extract_graph_data(window_sig, window_troughs, config=config, tol_samples=75)

    print(f"  PyG Data: x={list(data.x.shape)}, edges={data.edge_index.shape[1]}, "
          f"edge_attr={list(data.edge_attr.shape) if data.edge_attr is not None else 'None'}")
    print(f"  in_dim={data.x.shape[1]}, boundary nodes: {int(data.y.sum())}/{len(data.y)}")
    assert data.x.shape[0] > 0, "no nodes in graph"
    assert data.edge_index.shape[1] > 0, "no edges in graph"
    assert data.y.sum() > 0, "no boundary nodes labeled"
    results['features'] = True

    # 4. Forward pass through multiple architectures
    print("[4/6] Forward pass (GAT, GCN, SAGE, GIN)...")
    import torch
    from model import build_model

    device = 'cpu'  # sanity runs on CPU
    in_dim = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 0

    for arch in ['gat', 'gcn', 'sage', 'gin']:
        model_config = {
            'arch': arch, 'in_dim': in_dim, 'edge_dim': edge_dim,
            'hidden_dim': 32, 'n_heads': 4, 'n_layers': 2,
            'dropout': 0.0, 'use_rate_head': True, 'use_type_head': False,
        }
        model = build_model(model_config).to(device)
        out = model(data.x.to(device), data.edge_index.to(device),
                     edge_attr=data.edge_attr.to(device) if data.edge_attr is not None else None)
        assert 'boundary_logits' in out, f"{arch}: missing boundary_logits"
        assert out['boundary_logits'].shape[0] == data.x.shape[0], f"{arch}: wrong shape"
        print(f"  {arch}: logits shape={list(out['boundary_logits'].shape)} ✓")
    results['forward'] = True

    # 5. Compute boundary_f1 with dummy predictions
    print("[5/6] Computing boundary F1...")
    from train import compute_boundary_f1, compute_rate_mae

    # Use model predictions
    with torch.no_grad():
        logits = out['boundary_logits'].cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    node_bars = data.node_bars.numpy()
    pred_troughs = node_bars[scores > 0.5]
    gt_t = node_bars[data.y.numpy() > 0.5]

    f1 = compute_boundary_f1(pred_troughs, gt_t, tol_samples=75)
    print(f"  F1={f1:.4f} (random model, expected ~0)")
    results['metrics'] = True

    # 6. Config loading test
    print("[6/6] Config loading...")
    import yaml
    campaign = yaml.safe_load(open(Path(__file__).resolve().parent.parent / 'ml_metaopt_campaign.yaml'))
    assert campaign['campaign_id'] == 'dg-gat-respiratory-v2'
    assert campaign['version'] == 3
    results['config'] = True

    dt = time.time() - t0
    print(f"\n{'='*50}")
    print(f"All {len(results)} checks passed in {dt:.1f}s")
    assert dt < 60, f"Sanity check too slow: {dt:.1f}s > 60s"
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Fast mode (default)')
    args = parser.parse_args()

    try:
        ok = run_sanity(fast=True)
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"\nSANITY FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
