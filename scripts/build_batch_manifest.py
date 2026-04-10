"""Build delegated batch manifests for ray-hetzner queue execution.

Pure functions — no Ray, no network, no side effects.
Turns a list of trial configs + dispatch policy into a manifest dict
that the remote execution path can consume directly.
"""

from __future__ import annotations

import argparse
import json
import sys

from scripts.dispatch_policy import DispatchPolicy, classify_stage, resources_for_stage


def build_batch_manifest(
    campaign_id: str,
    iteration: int,
    trial_configs: list[dict],
    policy: DispatchPolicy,
    entrypoint: str = "python3 /root/dg_bidmc/scripts/ray_runner.py",
) -> dict:
    """Build a batch manifest from trial configs and dispatch policy.

    Returns a manifest dict with keys: batch_id, campaign_id, iteration, trials.
    Each trial record contains: trial_id, config, entrypoint, resources.
    Resources use only generic Ray keys (num_cpus, num_gpus) — never named hosts.
    """
    trials = []
    for config in trial_configs:
        stage = classify_stage(config)
        trials.append(
            {
                "trial_id": config["trial_id"],
                "config": config,
                "entrypoint": entrypoint,
                "resources": resources_for_stage(stage, policy),
            }
        )

    return {
        "batch_id": f"{campaign_id}-iter{iteration}-batch1",
        "campaign_id": campaign_id,
        "iteration": iteration,
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
        default="python3 /root/dg_bidmc/scripts/ray_runner.py",
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
