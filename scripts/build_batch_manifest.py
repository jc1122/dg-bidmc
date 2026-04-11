"""Build delegated batch manifests for ray-hetzner queue execution.

Pure functions — no Ray, no network, no side effects.
Turns a list of trial configs + dispatch policy into a manifest dict
that the remote execution path can consume directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

try:
    from scripts.dispatch_policy import DispatchPolicy, classify_stage, resources_for_stage
except ImportError:
    from dispatch_policy import DispatchPolicy, classify_stage, resources_for_stage


DEFAULT_ENTRYPOINT = "/home/jakub/ray-venv/bin/python3 scripts/ray_runner.py"
DEFAULT_METRICS_FILE = "metrics.json"


def build_batch_manifest(
    campaign_id: str,
    iteration: int,
    trial_configs: list[dict],
    policy: DispatchPolicy,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    artifact_path: Path | None = None,
) -> dict:
    """Build a batch manifest from trial configs and dispatch policy.

    Returns a manifest that is compatible with the shared ray-hetzner queue backend,
    while preserving the historical "trials" metadata block for local tooling/tests.
    """
    trials = []
    for config in trial_configs:
        stage = classify_stage(config)
        trials.append(
            {
                "trial_id": config["trial_id"],
                "config": copy.deepcopy(config),
                "entrypoint": entrypoint,
                "resources": resources_for_stage(stage, policy),
                }
            )

    if not trials:
        raise ValueError("trial_configs must contain at least one trial")

    primary_trial = trials[0]
    if artifact_path is None:
        artifact_path = Path("code.tar.gz")

    return {
        "version": 1,
        "batch_id": f"{campaign_id}-iter{iteration}-batch1",
        "campaign_id": campaign_id,
        "iteration": iteration,
        "experiment": {
            "trial_id": primary_trial["trial_id"],
            "dispatch_stage": primary_trial["config"].get("dispatch_stage"),
            "trial_count": len(trials),
        },
        "artifacts": {
            "code_artifact": {
                "uri": f"file://{artifact_path}",
            }
        },
        "execution": {
            "entrypoint": entrypoint,
            "env": {
                "METAOPT_EXPERIMENT_CONFIG_JSON": json.dumps(primary_trial["config"], sort_keys=True),
            },
        },
        "results_contract": {
            "metrics_file": DEFAULT_METRICS_FILE,
        },
        "retry_policy": {
            "max_attempts": 2,
        },
        "trials": trials,
    }


def _cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a delegated batch manifest from a JSON config list."
    )
    parser.add_argument("config_file", help="Path to JSON file with list of trial configs")
    parser.add_argument("--campaign-id", default="local", help="Campaign identifier")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number")
    parser.add_argument(
        "--entrypoint",
        default=DEFAULT_ENTRYPOINT,
        help="Entrypoint command for each trial",
    )
    args = parser.parse_args(argv)

    with open(args.config_file) as f:
        configs = json.load(f)

    policy = DispatchPolicy(
        cpu_screen_max_concurrent=6,
        cpu_screen_cpus_per_trial=4,
        gpu_promotion_max_concurrent=1,
        gpu_promotion_gpus_per_trial=1,
        cpu_eval_max_concurrent=4,
    )

    manifest = build_batch_manifest(
        args.campaign_id, args.iteration, configs, policy, args.entrypoint
    )
    json.dump(manifest, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    _cli_main()
