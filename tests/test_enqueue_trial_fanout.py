import importlib
import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module():
    import enqueue_trial_fanout

    return importlib.reload(enqueue_trial_fanout)


def test_build_manifest_embeds_trial_config_env(tmp_path):
    module = _load_module()
    artifact_path = tmp_path / "code.tar.gz"
    artifact_path.write_text("artifact")

    manifest = module.build_manifest(
        campaign_id="dg-gat-respiratory-v2",
        iteration=6,
        batch_id="iter6-screen-001",
        artifact_path=artifact_path,
        entrypoint=module.DEFAULT_ENTRYPOINT,
        trial_config={"dispatch_stage": "screen_cpu", "lr": 0.01},
        experiment={"submission": "screen", "trial_index": 1},
        retry_max_attempts=2,
    )

    assert manifest["artifacts"]["code_artifact"]["uri"] == f"file://{artifact_path}"
    assert manifest["results_contract"] == {"metrics_file": "metrics.json"}
    assert json.loads(manifest["execution"]["env"]["METAOPT_EXPERIMENT_CONFIG_JSON"]) == {
        "dispatch_stage": "screen_cpu",
        "lr": 0.01,
    }


def test_default_entrypoint_uses_aorus_ray_venv():
    module = _load_module()

    assert module.DEFAULT_ENTRYPOINT == "/home/jakub/ray-venv/bin/python3 scripts/ray_runner.py"
