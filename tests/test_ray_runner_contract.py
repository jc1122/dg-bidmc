import importlib
import json
import os
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module():
    import ray_runner

    return importlib.reload(ray_runner)


def test_main_writes_metrics_contract_from_experiment_env(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("METAOPT_WORKSPACE", str(workspace))
    monkeypatch.setenv(
        "METAOPT_EXPERIMENT_CONFIG_JSON",
        json.dumps({"dispatch_stage": "screen_cpu", "lr": 0.02}),
    )
    monkeypatch.delenv("METAOPT_TRIAL_CONFIG", raising=False)
    monkeypatch.delenv("METAOPT_RESULT_PATH", raising=False)

    module = _load_module()
    observed = {}

    def fake_run_trial(config):
        observed.update(config)
        return {
            "boundary_f1_600ms": 0.73,
            "rate_mae_bpm": 1.25,
            "by_dataset": {"bidmc_val": 0.73},
            "status": "SUCCESS",
        }

    monkeypatch.setattr(module, "run_trial", fake_run_trial)

    assert module.main() == 0
    assert observed["lr"] == 0.02

    metrics_path = workspace / "results" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["best_result"]["aggregate_metric"] == 0.73
    assert metrics["best_result"]["by_dataset"] == {"bidmc_val": 0.73}
