#!/usr/bin/env python3
"""Ray remote training runner for DG-GNN metaoptimization.

Reads experiment config from METAOPT_TRIAL_CONFIG env var (JSON string).
Writes results to METAOPT_RESULT_PATH.

Usage (standalone):
    METAOPT_TRIAL_CONFIG='{"arch":"gat","hidden_dim":64,...}' \
    METAOPT_RESULT_PATH='result.json' \
    python3 scripts/ray_runner.py

Called by metaopt queue infrastructure on the head node.
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Project root: prefer env override, then cwd, then fallback
PROJECT_ROOT = Path(os.environ.get('DG_PROJECT_ROOT', os.getcwd()))

def get_default_config():
    """Default config for a basic GAT experiment."""
    return {
        # Graph construction
        'graph_variant': 'standard',
        'lp_cutoff_hz': 2.0,
        'n_levels': 3,
        'detrend': 'none',
        'burst_suppress': False,
        'wavelet_denoise': False,

        # Features (always on)
        'feat_bar_position': True,
        'feat_log_edge_size': True,
        'feat_is_low': True,

        # Features (optional - defaults)
        'feat_amplitude': True,
        'feat_duration': True,
        'feat_level': True,
        'feat_node_degree': False,
        'feat_commitment_ratio': False,
        'feat_run_asymmetry': False,
        'feat_span_overlap': False,
        'feat_swing_velocity': False,
        'feat_birth_rate': False,
        'feat_edge_size_ratio': False,
        'feat_amplitude_delta': False,
        'feat_duration_ratio': False,
        'feat_phase_estimate': False,

        # Edge features
        'edge_feat_log_size': True,
        'edge_feat_direction_match': True,
        'edge_feat_direction': False,
        'edge_feat_duration_norm': False,
        'edge_feat_amplitude_delta': False,

        # Architecture
        'arch': 'gat',
        'hidden_dim': 64,
        'n_heads': 8,
        'n_layers': 3,
        'dropout': 0.1,
        'residual': False,
        'batch_norm': False,
        'concat_heads': True,
        'aggr': 'mean',
        'boundary_head_layers': 1,
        'use_rate_head': True,
        'use_type_head': False,

        # Training
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'pos_weight': 7.0,
        'loss_rate_weight': 0.0,
        'loss_type_weight': 0.0,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 0,
        'max_epochs': 100,
        'patience': 15,
        'window_breaths': 6,
        'tol_samples_train': 75,
        'batch_size': 32,
        'augment': 'none',
        'augment_prob': 0.5,
        'n_augmented_copies': 0,
        'augment_snr_min': 15.0,
        'augment_snr_max': 30.0,
        'augment_drift_min': 0.0005,
        'augment_drift_max': 0.003,
        'augment_max_simultaneous': 3,
        'label_sigma': 0,  # 0 = hard labels; >0 = Gaussian soft labels (samples)
    }

def run_trial(config: dict) -> dict:
    """Run one training trial. Returns result dict.
    
    GPU-robust: catches HIP/CUDA errors and retries on CPU if needed.
    Supports ensemble mode via config['n_ensemble'] > 1.
    """
    n_ensemble = config.get('n_ensemble', 1)
    if n_ensemble > 1:
        from train import train_ensemble as train_fn
    else:
        from train import train as train_fn

    # Data directories on the worker
    train_dir = str(PROJECT_ROOT / 'data' / 'graphs' / 'train')
    val_dir = str(PROJECT_ROOT / 'data' / 'graphs' / 'val')
    val_adv_dir = str(PROJECT_ROOT / 'data' / 'graphs' / 'val_adversarial')

    # Ensure data directories exist (train() rebuilds graphs if empty)
    for d in [train_dir, val_dir, val_adv_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if not Path(val_adv_dir).exists():
        val_adv_dir = None

    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
    except ImportError:
        pass

    config['_original_device'] = device
    t0 = time.time()
    
    # Try GPU first, fall back to CPU on fatal GPU error
    max_gpu_retries = 2
    for attempt in range(max_gpu_retries + 1):
        try:
            result = train_fn(config, train_dir, val_dir, val_adv_dir,
                         device=device, verbose=True)
            break
        except RuntimeError as e:
            err_msg = str(e).lower()
            is_gpu_err = any(k in err_msg for k in (
                'hip', 'cuda', 'device-side assert',
                'invalid device function', 'out of memory',
            ))
            if is_gpu_err and device != 'cpu' and attempt < max_gpu_retries:
                print(f"  [GPU CRASH attempt {attempt+1}] {e}")
                import torch
                torch.cuda.empty_cache()
                if attempt == max_gpu_retries - 1:
                    print("  Falling back to CPU training")
                    device = 'cpu'
            else:
                raise
    
    wall_time = time.time() - t0

    # Format output for metaopt
    best = result.get('best_metrics', {})
    output = {
        'boundary_f1_600ms': best.get('boundary_f1_600ms', 0.0),
        'rate_mae_bpm': best.get('rate_mae_bpm', float('inf')),
        'by_dataset': {
            'bidmc_val': best.get('val_f1', 0.0),
        },
        'status': 'SUCCESS',
        'wall_time_seconds': wall_time,
        'n_epochs_run': result.get('n_epochs_run', 0),
        'n_params': result.get('n_params', 0),
    }
    if 'val_adv_f1' in best:
        output['by_dataset']['bidmc_val_adversarial'] = best['val_adv_f1']

    return output

def main():
    # Read config from environment
    config_json = os.environ.get('METAOPT_TRIAL_CONFIG', '{}')
    result_path = os.environ.get('METAOPT_RESULT_PATH', 'result.json')

    # Merge with defaults
    config = get_default_config()
    try:
        trial_config = json.loads(config_json)
        config.update(trial_config)
    except json.JSONDecodeError as e:
        result = {'boundary_f1_600ms': 0.0, 'error': f'Invalid config JSON: {e}', 'status': 'FAILED'}
        Path(result_path).write_text(json.dumps(result, indent=2))
        sys.exit(1)

    print(f"Running trial with config: {json.dumps(config, indent=2)}")

    try:
        result = run_trial(config)
    except Exception as e:
        result = {
            'boundary_f1_600ms': 0.0,
            'rate_mae_bpm': float('inf'),
            'error': str(e),
            'traceback': traceback.format_exc(),
            'status': 'FAILED',
        }

    # Write result
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(result_path).write_text(json.dumps(result, indent=2))
    print(f"\nResult written to {result_path}")
    print(f"  boundary_f1_600ms: {result.get('boundary_f1_600ms', 'N/A')}")
    print(f"  rate_mae_bpm: {result.get('rate_mae_bpm', 'N/A')}")
    print(f"  status: {result.get('status', 'UNKNOWN')}")

if __name__ == '__main__':
    main()
