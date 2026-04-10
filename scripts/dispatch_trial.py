#!/usr/bin/env python3
"""Robust Ray trial dispatcher for DG-GNN metaoptimization.

Supports:
- Stage-aware dispatch: screen_cpu -> train_gpu -> eval_cpu
- Generic resource requirements (num_cpus / num_gpus only -- no named hosts)
- GPU training with crash recovery + CPU fallback
- CPU-only training on head node (16 cores)
- Retry logic for transient GPU errors
- Dry-run manifest generation (no Ray contact)
- runtime_env packaging with .gitignore bypass
- Concurrent-safe: each trial gets its own runtime_env sandbox

Usage (on head node):
    source /opt/ray-env/bin/activate
    python3 scripts/dispatch_trial.py --config .ml-metaopt/trial_config.json --out result.json
    python3 scripts/dispatch_trial.py --config '{"arch":"gat","hidden_dim":64}' --out result.json --cpu-only
    python3 scripts/dispatch_trial.py --config trial.json --out manifest.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


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


def _resolve_stage(config: dict, cpu_only: bool = False) -> str:
    """Determine dispatch stage from config and flags.

    Priority: cpu_only flag forces screen_cpu, else use config dispatch_stage.
    """
    try:
        from scripts.dispatch_policy import classify_stage
    except ImportError:
        from dispatch_policy import classify_stage
    if cpu_only:
        return "screen_cpu"
    return classify_stage(config)


def _build_dry_run_manifest(config: dict, stage: str, output_path: Path) -> dict:
    """Build and persist a dry-run manifest without contacting Ray."""
    try:
        from scripts.dispatch_policy import (
            DispatchPolicy, resources_for_stage, prepare_stage_config,
        )
    except ImportError:
        from dispatch_policy import (
            DispatchPolicy, resources_for_stage, prepare_stage_config,
        )

    policy = DispatchPolicy()
    resources = resources_for_stage(stage, policy)
    shaped_config = prepare_stage_config(config, stage)

    manifest = {
        "dry_run": True,
        "dispatch_stage": stage,
        "resources": resources,
        "config": shaped_config,
        "trial_id": config.get("trial_id", "dry-run-0"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = json.dumps(manifest, indent=2)
    output_path.write_text(manifest_json)
    print(manifest_json)
    return manifest


def _run_trial_remote(config_json: str) -> str:
    """Run a trial inside a Ray worker (CPU or GPU -- resource-agnostic).

    Expects *config_json* to be **already shaped** by
    ``dispatch_policy.prepare_stage_config`` — no re-shaping is done here.
    """
    import json as _json
    import os as _os
    import sys as _sys
    import traceback as _tb

    cwd = _os.getcwd()
    _sys.path.insert(0, _os.path.join(cwd, "scripts"))
    _os.environ["DG_PROJECT_ROOT"] = cwd
    try:
        from ray_runner import run_trial, get_default_config

        config = _json.loads(config_json)
        full = get_default_config()
        full.update(config)
        return _json.dumps(run_trial(full))
    except Exception as e:
        return _json.dumps({
            "error": str(e),
            "traceback": _tb.format_exc(),
            "boundary_f1_600ms": 0.0,
            "status": "FAILED",
        })


def dispatch(config: dict, project_root: Path, output_path: Path,
             cpu_only: bool = False, gpu_retries: int = 2,
             dry_run: bool = False):
    """Submit trial with stage-aware resource selection and retry logic."""
    stage = _resolve_stage(config, cpu_only=cpu_only)

    if dry_run:
        return _build_dry_run_manifest(config, stage, output_path)

    import ray
    try:
        from scripts.dispatch_policy import (
            DispatchPolicy, resources_for_stage, prepare_stage_config,
        )
    except ImportError:
        from dispatch_policy import (
            DispatchPolicy, resources_for_stage, prepare_stage_config,
        )

    policy = DispatchPolicy()
    resources = resources_for_stage(stage, policy)

    hide_gitignore(project_root)
    try:
        runtime = make_runtime_env(project_root)
        ray.init(address="auto", runtime_env=runtime)
    finally:
        restore_gitignore(project_root)

    # Shape once — deepcopy, so the caller's dict is never mutated
    shaped = prepare_stage_config(config, stage)
    config_json = json.dumps(shaped)

    print(f"Dispatching trial (stage={stage}, resources={resources}, "
          f"{len(shaped)} config keys)")

    # Build a Ray remote with the resolved generic resources
    remote_fn = ray.remote(**resources)(_run_trial_remote)

    t0 = time.time()
    is_gpu = stage == "train_gpu"

    if not is_gpu:
        ref = remote_fn.remote(config_json)
        result_json = ray.get(ref)
    else:
        last_err = None
        result_json = None
        for attempt in range(gpu_retries + 1):
            try:
                ref = remote_fn.remote(config_json)
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
                    cpu_resources = resources_for_stage("screen_cpu", policy)
                    cpu_fn = ray.remote(**cpu_resources)(_run_trial_remote)
                    ref = cpu_fn.remote(config_json)
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
                cpu_resources = resources_for_stage("screen_cpu", policy)
                cpu_fn = ray.remote(**cpu_resources)(_run_trial_remote)
                ref = cpu_fn.remote(config_json)
                result_json = ray.get(ref, timeout=3600)
            except ray.exceptions.GetTimeoutError:
                print(f"  Trial timed out on attempt {attempt+1}")
                result_json = json.dumps({
                    "error": "timeout",
                    "boundary_f1_600ms": 0.0,
                    "status": "FAILED",
                })
                break

    elapsed = time.time() - t0
    result = json.loads(result_json)
    result["dispatch_wall_time"] = elapsed
    result["dispatch_stage"] = stage

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
    parser = argparse.ArgumentParser(
        description="Dispatch a DG-GNN trial via Ray with stage-aware resources."
    )
    parser.add_argument("--config", required=True,
                        help="JSON file path or inline JSON string")
    parser.add_argument("--out", required=True, help="Output result JSON path")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force screen_cpu stage regardless of config")
    parser.add_argument("--gpu-retries", type=int, default=2)
    parser.add_argument("--project-root", default="/root/dg_bidmc")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print and write manifest JSON without contacting Ray")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = json.loads(Path(args.config).read_text())
    else:
        config = json.loads(args.config)

    dispatch(config, Path(args.project_root), Path(args.out),
             cpu_only=args.cpu_only, gpu_retries=args.gpu_retries,
             dry_run=args.dry_run)


if __name__ == "__main__":
    main()
