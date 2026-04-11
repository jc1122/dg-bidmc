#!/usr/bin/env python3
"""P92: Soft-F1 loss experiment.
3 configs × 3 seeds = 9 runs. Compare bce, soft_f1, bce+soft_f1."""
import sys, os, json, traceback, numpy as np
os.chdir('/home/jakub/projects/dg_bidmc')
sys.path.insert(0, 'scripts')

import torch
from train import train, evaluate, load_cached_graphs, optimize_threshold, compute_feature_dims
from torch_geometric.loader import DataLoader
from model import build_model

TRAIN_DIR = 'data/graphs/train'
VAL_DIR = 'data/graphs/val'
VAL_ADV_DIR = 'data/graphs/val_adversarial'
TEST_DIR = 'data/graphs/test'

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

BASE = {
    'arch': 'transformer', 'n_layers': 5, 'hidden_dim': 128, 'n_heads': 4,
    'dropout': 0.1, 'weight_decay': 1e-4, 'lr': 1e-3, 'optimizer': 'adamw',
    'scheduler': 'cosine', 'max_epochs': 100, 'patience': 15, 'batch_size': 32,
    'add_self_loops': False,
}

TRIALS = {
    "bce-baseline": {**BASE, 'loss_fn': 'bce'},
    "soft-f1-only": {**BASE, 'loss_fn': 'soft_f1'},
    "bce+soft-f1-a0.3": {**BASE, 'loss_fn': 'bce+soft_f1', 'soft_f1_alpha': 0.3},
    "bce+soft-f1-a0.5": {**BASE, 'loss_fn': 'bce+soft_f1', 'soft_f1_alpha': 0.5},
}

SEEDS = [42, 123, 456]
ROBUST_PP = {'threshold': 0.15, 'nms_dist': 250}

all_results = {}
for trial_name, config in TRIALS.items():
    trial_results = []
    for seed in SEEDS:
        run_name = f"{trial_name}_s{seed}"
        print(f"\n{'='*60}")
        print(f"TRIAL: {run_name}")
        print(f"{'='*60}")
        set_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            result = train(config, TRAIN_DIR, VAL_DIR, VAL_ADV_DIR, device=device, verbose=True)
            in_dim, edge_dim = compute_feature_dims(config)
            model_cfg = {**config, 'in_dim': in_dim, 'edge_dim': edge_dim}
            model = build_model(model_cfg).to(device)
            ckpt = 'data/graphs/checkpoints/best_model.pt'
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()
            val_data = load_cached_graphs(VAL_DIR)
            test_data = load_cached_graphs(TEST_DIR)
            val_loader = DataLoader(val_data, batch_size=32)
            test_loader = DataLoader(test_data, batch_size=32)
            pp = optimize_threshold(model, val_loader, device, tol_samples=75)
            val_r = evaluate(model, val_loader, device, tol_samples=75,
                            threshold=pp[0], nms_dist=pp[1])
            test_r = evaluate(model, test_loader, device, tol_samples=75,
                             threshold=pp[0], nms_dist=pp[1])
            test_robust = evaluate(model, test_loader, device, tol_samples=75,
                                  threshold=ROBUST_PP['threshold'], nms_dist=ROBUST_PP['nms_dist'])
            test_raw = evaluate(model, test_loader, device, tol_samples=75,
                               threshold=0.5, nms_dist=0)
            r = {
                'run_name': run_name, 'seed': seed, 'status': 'ok',
                'val_pp': val_r['boundary_f1_600ms'],
                'test_valpp': test_r['boundary_f1_600ms'],
                'test_robust': test_robust['boundary_f1_600ms'],
                'test_raw': test_raw['boundary_f1_600ms'],
                'epoch': result.get('best_epoch', -1),
            }
            trial_results.append(r)
            print(f"  RESULT: val_pp={r['val_pp']:.4f} test_valpp={r['test_valpp']:.4f} "
                  f"test_robust={r['test_robust']:.4f} test_raw={r['test_raw']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            trial_results.append({'run_name': run_name, 'seed': seed, 'status': 'error', 'error': str(e)})
    all_results[trial_name] = trial_results

print("\n" + "="*80)
print("P92 SOFT-F1 LOSS SUMMARY")
print("="*80)
print(f"{'Trial':>20s}  {'Test Robust':>12s}  {'Test Raw':>10s}  {'Seeds':>6s}")
print("-"*60)
for name, runs in all_results.items():
    ok = [r for r in runs if r['status'] == 'ok']
    if ok:
        rob = np.mean([r['test_robust'] for r in ok])
        raw = np.mean([r['test_raw'] for r in ok])
        std = np.std([r['test_robust'] for r in ok])
        print(f"{name:>20s}  {rob:.4f}±{std:.4f}  {raw:10.4f}  {len(ok):>6d}")
    else:
        print(f"{name:>20s}  ALL FAILED")

os.makedirs('results/aorus', exist_ok=True)
with open('results/aorus/p92_soft_f1.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to results/aorus/p92_soft_f1.json")
