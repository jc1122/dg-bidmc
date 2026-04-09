"""Evaluation harness for DG-GNN respiratory boundary detection.

Compares trained model against baselines on test set.
Reports boundary F1 @600ms and respiratory rate MAE.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from model import build_model
from train import load_cached_graphs, evaluate as eval_model, compute_boundary_f1, compute_rate_mae
from graph_features import compute_feature_dims

def evaluate_model_on_split(
    model_path: str,
    config: dict,
    data_dir: str,
    device: str = 'cuda',
) -> dict:
    """Load a saved model and evaluate on a data split."""
    in_dim, edge_dim = compute_feature_dims(config)
    model_config = {**config, 'in_dim': in_dim, 'edge_dim': edge_dim}
    model = build_model(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    data = load_cached_graphs(data_dir)
    loader = DataLoader(data, batch_size=32)
    return eval_model(model, loader, device)

def evaluate_all(
    model_path: str,
    config: dict,
    test_dir: str,
    test_adv_dir: str | None = None,
    device: str = 'cuda',
) -> dict:
    """Full evaluation with per-condition breakdown."""
    results = {}
    results['test'] = evaluate_model_on_split(model_path, config, test_dir, device)
    if test_adv_dir:
        results['test_adversarial'] = evaluate_model_on_split(model_path, config, test_adv_dir, device)
    
    # Aggregate
    if test_adv_dir:
        results['aggregate_f1'] = 0.55 * results['test']['boundary_f1_600ms'] + 0.45 * results['test_adversarial']['boundary_f1_600ms']
    else:
        results['aggregate_f1'] = results['test']['boundary_f1_600ms']
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model state dict')
    parser.add_argument('--config', required=True, help='Path to config JSON')
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--test-adv-dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    config = json.loads(Path(args.config).read_text())
    results = evaluate_all(args.model, config, args.test_dir, args.test_adv_dir, args.device)
    
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
