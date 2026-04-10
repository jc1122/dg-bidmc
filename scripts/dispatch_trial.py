#!/usr/bin/env python3
"""Robust Ray trial dispatcher for DG-GNN metaoptimization.

Supports:
- GPU training on Aorus with crash recovery + CPU fallback
- CPU-only training on head node (16 cores)
- Retry logic for transient GPU errors
- runtime_env packaging with .gitignore bypass
- Concurrent-safe: each trial gets its own runtime_env sandbox

Usage (on head node):
    source /opt/ray-env/bin/activate
    python3 scripts/dispatch_trial.py --config .ml-metaopt/trial_config.json --out result.json
    python3 scripts/dispatch_trial.py --config '{"arch":"gat","hidden_dim":64}' --out result.json --cpu-only
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
import ray


def hide_gitignore(project_root: Path):
    """Temporarily hide .gitignore so Ray packages all data files."""
    gi = project_root / '.gitignore'
    bak = project_root / '.gitignore.bak'
    if gi.exists():
        shutil.move(str(gi), str(bak))
    return bak


def restore_gitignore(project_root: Path):
    bak = project_root / '.gitignore.bak'
    gi = project_root / '.gitignore'
    if bak.exists():
        shutil.move(str(bak), str(gi))


def make_runtime_env(project_root: Path) -> dict:
    return {
        "working_dir": str(project_root),
        "excludes": [
            ".git", "__pycache__", "*.pyc",
            ".ml-metaopt", "notebooks", "results",
            "data/bidmc", "data/graphs", ".gitignore.bak",
        ],
    }


@ray.remote
def run_on_cpu(config_json: str) -> str:
    """CPU-only training on head node."""
    import json, os, sys, traceback
    cwd = os.getcwd()
    sys.path.insert(0, os.path.join(cwd, "scripts"))
    config = json.loads(config_json)
    os.environ["DG_PROJECT_ROOT"] = cwd
    try:
        from ray_runner import run_trial, get_default_config
        full = get_default_config()
        full.update(config)
        return json.dumps(run_trial(full))
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc(),
                           "boundary_f1_600ms": 0.0, "status": "FAILED"})


@ray.remote(num_gpus=1, resources={"aorus": 1})
def run_on_gpu(config_json: str) -> str:
    """GPU training on Aorus. Falls back to CPU internally on HIP errors."""
    import json, os, sys, traceback
    cwd = os.getcwd()
    sys.path.insert(0, os.path.join(cwd, "scripts"))
    config = json.loads(config_json)
    os.environ["DG_PROJECT_ROOT"] = cwd
    try:
        from ray_runner import run_trial, get_default_config
        full = get_default_config()
        full.update(config)
        return json.dumps(run_trial(full))
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc(),
                           "boundary_f1_600ms": 0.0, "status": "FAILED"})


def dispatch(config: dict, project_root: Path, output_path: Path,
             cpu_only: bool = False, gpu_retries: int = 2):
    """Submit trial with retry logic."""
    hide_gitignore(project_root)
    try:
        runtime = make_runtime_env(project_root)
        ray.init(address="auto", runtime_env=runtime)
    finally:
        restore_gitignore(project_root)

    config_json = json.dumps(config)
    print(f"Dispatching trial ({len(config)} keys, cpu_only={cpu_only})")

    t0 = time.time()
    if cpu_only:
        ref = run_on_cpu.remote(config_json)
        result_json = ray.get(ref)
    else:
        last_err = None
        result_json = None
        for attempt in range(gpu_retries + 1):
            try:
                ref = run_on_gpu.remote(config_json)
                result_json = ray.get(ref, timeout=1800)  # 30 min max
                result = json.loads(result_json)
                if result.get("status") != "FAILED":
                    break
                err = result.get("error", "")
                if any(k in err.lower() for k in ('hip', 'cuda', 'device', 'out of memory')):
                    last_err = err
                    print(f"  GPU attempt {attempt+1}/{gpu_retries+1} failed: {err[:120]}")
                    if attempt < gpu_retries:
                        time.sleep(5)
                        continue
                    # Final fallback: CPU
                    print("  All GPU attempts failed, falling back to CPU on head")
                    ref = run_on_cpu.remote(config_json)
                    result_json = ray.get(ref, timeout=3600)
                else:
                    break
            except ray.exceptions.RayTaskError as e:
                last_err = str(e)
                print(f"  Ray task error attempt {attempt+1}: {str(e)[:200]}")
                if attempt < gpu_retries:
                    time.sleep(5)
                    continue
                print("  Falling back to CPU")
                ref = run_on_cpu.remote(config_json)
                result_json = ray.get(ref, timeout=3600)
            except ray.exceptions.GetTimeoutError:
                print(f"  Trial timed out on attempt {attempt+1}")
                result_json = json.dumps({"error": "timeout", "boundary_f1_600ms": 0.0, "status": "FAILED"})
                break

    elapsed = time.time() - t0
    result = json.loads(result_json)
    result["dispatch_wall_time"] = elapsed

    # Print summary
    print(f"\nCompleted in {elapsed:.1f}s")
    if "error" in result:
        print(f"ERROR: {result['error'][:200]}")
    else:
        print(f"boundary_f1_600ms: {result.get('boundary_f1_600ms', 'N/A')}")
        print(f"rate_mae_bpm: {result.get('rate_mae_bpm', 'N/A')}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Result saved to {output_path}")

    ray.shutdown()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON file path or inline JSON string")
    parser.add_argument("--out", required=True, help="Output result JSON path")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu-retries", type=int, default=2)
    parser.add_argument("--project-root", default="/root/dg_bidmc")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = json.loads(Path(args.config).read_text())
    else:
        config = json.loads(args.config)

    dispatch(config, Path(args.project_root), Path(args.out),
             cpu_only=args.cpu_only, gpu_retries=args.gpu_retries)


if __name__ == "__main__":
    main()
