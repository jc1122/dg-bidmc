"""Tests for scripts/build_batch_manifest.py."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.build_batch_manifest import build_batch_manifest
from scripts.dispatch_policy import DispatchPolicy


@pytest.fixture()
def policy() -> DispatchPolicy:
    return DispatchPolicy(
        cpu_screen_max_concurrent=6,
        cpu_screen_cpus_per_trial=4,
        gpu_promotion_max_concurrent=1,
        gpu_promotion_gpus_per_trial=1,
        cpu_eval_max_concurrent=4,
    )


# --- core behaviour ---


def test_multiple_screen_cpu_trials(policy: DispatchPolicy) -> None:
    configs = [
        {"trial_id": "s1", "dispatch_stage": "screen_cpu", "lr": 1e-3},
        {"trial_id": "s2", "dispatch_stage": "screen_cpu", "lr": 2e-3},
        {"trial_id": "s3", "dispatch_stage": "screen_cpu", "lr": 3e-3},
    ]
    manifest = build_batch_manifest("camp1", 0, configs, policy)

    assert len(manifest["trials"]) == 3
    ids = [t["trial_id"] for t in manifest["trials"]]
    assert ids == ["s1", "s2", "s3"]
    for trial in manifest["trials"]:
        assert trial["resources"] == {"num_cpus": 4}


def test_train_gpu_trial_emits_gpu(policy: DispatchPolicy) -> None:
    configs = [{"trial_id": "g1", "dispatch_stage": "train_gpu", "lr": 1e-4}]
    manifest = build_batch_manifest("camp1", 1, configs, policy)

    trial = manifest["trials"][0]
    assert trial["resources"] == {"num_gpus": 1}


def test_no_named_resources(policy: DispatchPolicy) -> None:
    """No named-host or named-resource like 'aorus' may appear."""
    configs = [
        {"trial_id": "s1", "dispatch_stage": "screen_cpu"},
        {"trial_id": "g1", "dispatch_stage": "train_gpu"},
        {"trial_id": "e1", "dispatch_stage": "eval_cpu"},
    ]
    manifest = build_batch_manifest("camp1", 2, configs, policy)

    blob = json.dumps(manifest)
    assert "aorus" not in blob.lower()
    for trial in manifest["trials"]:
        for key in trial["resources"]:
            assert key.startswith("num_"), f"Non-generic resource key: {key}"


# --- manifest metadata ---


def test_batch_id_format(policy: DispatchPolicy) -> None:
    manifest = build_batch_manifest(
        "mycamp", 3, [{"trial_id": "t1", "dispatch_stage": "screen_cpu"}], policy
    )
    assert manifest["batch_id"] == "mycamp-iter3-batch1"
    assert manifest["campaign_id"] == "mycamp"
    assert manifest["iteration"] == 3


def test_entrypoint_default(policy: DispatchPolicy) -> None:
    configs = [{"trial_id": "t1", "dispatch_stage": "screen_cpu"}]
    manifest = build_batch_manifest("c", 0, configs, policy)
    assert manifest["trials"][0]["entrypoint"] == "python3 /root/dg_bidmc/scripts/ray_runner.py"


def test_entrypoint_custom(policy: DispatchPolicy) -> None:
    configs = [{"trial_id": "t1", "dispatch_stage": "screen_cpu"}]
    manifest = build_batch_manifest("c", 0, configs, policy, entrypoint="custom_cmd")
    assert manifest["trials"][0]["entrypoint"] == "custom_cmd"


def test_config_preserved_in_trial(policy: DispatchPolicy) -> None:
    cfg = {"trial_id": "x1", "dispatch_stage": "train_gpu", "lr": 0.01, "layers": 3}
    manifest = build_batch_manifest("c", 0, [cfg], policy)
    assert manifest["trials"][0]["config"] == cfg


# --- CLI ---


def test_cli_prints_json(policy: DispatchPolicy) -> None:
    """The script should accept a JSON file and print a manifest."""
    import os
    import tempfile

    configs = [
        {"trial_id": "c1", "dispatch_stage": "screen_cpu"},
        {"trial_id": "c2", "dispatch_stage": "train_gpu"},
    ]
    # Write config to a file in the project directory
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "_test_cli_configs.json"
    )
    try:
        with open(cfg_path, "w") as f:
            json.dump(configs, f)

        result = subprocess.run(
            [sys.executable, "-m", "scripts.build_batch_manifest",
             "--campaign-id", "cli-test", "--iteration", "5", cfg_path],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0, result.stderr
        manifest = json.loads(result.stdout)
        assert manifest["batch_id"] == "cli-test-iter5-batch1"
        assert len(manifest["trials"]) == 2
    finally:
        if os.path.exists(cfg_path):
            os.remove(cfg_path)
