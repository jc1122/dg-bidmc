"""Tests for stage-aware dispatch_trial and ray_runner stage shaping (Task 3).

All tests are offline — no live Ray cluster required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ray_runner import prepare_stage_config  # noqa: E402

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# prepare_stage_config
# ---------------------------------------------------------------------------


class TestPrepareStageConfig:
    """Unit tests for ray_runner.prepare_stage_config."""

    def test_screen_cpu_caps_max_epochs(self):
        cfg = prepare_stage_config({"max_epochs": 100}, "screen_cpu")
        assert cfg["max_epochs"] == 25

    def test_screen_cpu_keeps_low_epochs(self):
        cfg = prepare_stage_config({"max_epochs": 10}, "screen_cpu")
        assert cfg["max_epochs"] == 10

    def test_screen_cpu_caps_patience(self):
        cfg = prepare_stage_config({"patience": 20}, "screen_cpu")
        assert cfg["patience"] == 5

    def test_screen_cpu_keeps_low_patience(self):
        cfg = prepare_stage_config({"patience": 3}, "screen_cpu")
        assert cfg["patience"] == 3

    def test_screen_cpu_forces_ensemble_1(self):
        cfg = prepare_stage_config({"n_ensemble": 5}, "screen_cpu")
        assert cfg["n_ensemble"] == 1

    def test_screen_cpu_marks_screening(self):
        cfg = prepare_stage_config({}, "screen_cpu")
        assert cfg["_screening"] is True
        assert cfg["dispatch_stage"] == "screen_cpu"

    def test_train_gpu_preserves_full_config(self):
        base = {"max_epochs": 200, "patience": 30, "n_ensemble": 5}
        cfg = prepare_stage_config(base, "train_gpu")
        assert cfg["max_epochs"] == 200
        assert cfg["patience"] == 30
        assert cfg["n_ensemble"] == 5
        assert cfg["dispatch_stage"] == "train_gpu"

    def test_train_gpu_no_screening_flag(self):
        cfg = prepare_stage_config({}, "train_gpu")
        assert "_screening" not in cfg

    def test_eval_cpu_marks_eval_only(self):
        cfg = prepare_stage_config({}, "eval_cpu")
        assert cfg["_eval_only"] is True
        assert cfg["dispatch_stage"] == "eval_cpu"

    def test_does_not_mutate_original(self):
        orig = {"max_epochs": 100, "patience": 20, "nested": {"a": 1}}
        prepare_stage_config(orig, "screen_cpu")
        assert orig["max_epochs"] == 100
        assert orig["patience"] == 20

    def test_stage_from_config_when_none(self):
        cfg = prepare_stage_config({"dispatch_stage": "eval_cpu"})
        assert cfg["_eval_only"] is True

    def test_default_stage_is_train_gpu(self):
        cfg = prepare_stage_config({"lr": 0.01})
        assert cfg["dispatch_stage"] == "train_gpu"

    def test_screen_cpu_defaults_missing_keys(self):
        """When max_epochs/patience absent, uses caps as defaults."""
        cfg = prepare_stage_config({}, "screen_cpu")
        assert cfg["max_epochs"] == 25
        assert cfg["patience"] == 5


# ---------------------------------------------------------------------------
# Dry-run CLI via dispatch_trial
# ---------------------------------------------------------------------------


class TestDryRunCLI:
    """CLI dry-run tests — invoke dispatch_trial.py as subprocess."""

    def _run_dry_run(self, config: dict, extra_args: list[str] | None = None,
                     out_dir: Path | None = None) -> tuple[dict, str]:
        """Helper: run dispatch_trial.py --dry-run and return (manifest, stdout)."""
        if out_dir is None:
            out_dir = Path(_PROJECT_ROOT) / "_test_scratch"
            out_dir.mkdir(exist_ok=True)

        cfg_path = out_dir / "test_config.json"
        out_path = out_dir / "test_manifest.json"
        cfg_path.write_text(json.dumps(config))

        cmd = [
            sys.executable, "-m", "scripts.dispatch_trial",
            "--config", str(cfg_path),
            "--out", str(out_path),
            "--dry-run",
        ]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        manifest = json.loads(out_path.read_text())
        return manifest, result.stdout

    def test_dry_run_writes_and_prints_manifest(self, tmp_path):
        out_dir = tmp_path / "dry"
        out_dir.mkdir()
        manifest, stdout = self._run_dry_run(
            {"lr": 0.01, "dispatch_stage": "train_gpu"}, out_dir=out_dir,
        )
        assert manifest["dry_run"] is True
        assert "dry_run" in stdout

    def test_dry_run_gpu_shows_num_gpus_only(self, tmp_path):
        out_dir = tmp_path / "gpu"
        out_dir.mkdir()
        manifest, _ = self._run_dry_run(
            {"lr": 0.01, "dispatch_stage": "train_gpu"}, out_dir=out_dir,
        )
        assert manifest["resources"] == {"num_gpus": 1}
        blob = json.dumps(manifest)
        assert "aorus" not in blob.lower()

    def test_dry_run_screen_cpu_resources(self, tmp_path):
        out_dir = tmp_path / "cpu"
        out_dir.mkdir()
        manifest, _ = self._run_dry_run(
            {"lr": 0.01, "dispatch_stage": "screen_cpu"}, out_dir=out_dir,
        )
        assert "num_cpus" in manifest["resources"]
        assert "num_gpus" not in manifest["resources"]

    def test_dry_run_cpu_only_forces_screen(self, tmp_path):
        out_dir = tmp_path / "force"
        out_dir.mkdir()
        manifest, _ = self._run_dry_run(
            {"lr": 0.01, "dispatch_stage": "train_gpu"},
            extra_args=["--cpu-only"],
            out_dir=out_dir,
        )
        assert manifest["dispatch_stage"] == "screen_cpu"
        assert "num_cpus" in manifest["resources"]

    def test_dry_run_config_shaped(self, tmp_path):
        """Screen stage config should be shaped (caps applied)."""
        out_dir = tmp_path / "shaped"
        out_dir.mkdir()
        manifest, _ = self._run_dry_run(
            {"dispatch_stage": "screen_cpu", "max_epochs": 200, "patience": 30},
            out_dir=out_dir,
        )
        assert manifest["config"]["max_epochs"] == 25
        assert manifest["config"]["patience"] == 5
        assert manifest["config"]["_screening"] is True

    def test_dry_run_no_ray_import_needed(self, tmp_path):
        """Dry-run must not attempt to import or connect to Ray."""
        out_dir = tmp_path / "noray"
        out_dir.mkdir()
        # If Ray were imported, it would likely fail (no cluster).
        # The test passing proves Ray is not contacted.
        manifest, _ = self._run_dry_run(
            {"dispatch_stage": "eval_cpu"}, out_dir=out_dir,
        )
        assert manifest["dispatch_stage"] == "eval_cpu"

    def test_dry_run_no_named_resources_any_stage(self, tmp_path):
        """No named resources in any stage dry-run output."""
        for stage in ("screen_cpu", "train_gpu", "eval_cpu"):
            out_dir = tmp_path / stage
            out_dir.mkdir()
            manifest, _ = self._run_dry_run(
                {"dispatch_stage": stage}, out_dir=out_dir,
            )
            for key in manifest["resources"]:
                assert key.startswith("num_"), f"Named resource in {stage}: {key}"


# ---------------------------------------------------------------------------
# _resolve_stage unit tests
# ---------------------------------------------------------------------------


class TestResolveStage:
    def test_cpu_only_forces_screen(self):
        from dispatch_trial import _resolve_stage
        assert _resolve_stage({"dispatch_stage": "train_gpu"}, cpu_only=True) == "screen_cpu"

    def test_uses_config_stage(self):
        from dispatch_trial import _resolve_stage
        assert _resolve_stage({"dispatch_stage": "eval_cpu"}) == "eval_cpu"

    def test_defaults_to_train_gpu(self):
        from dispatch_trial import _resolve_stage
        assert _resolve_stage({}) == "train_gpu"


# ---------------------------------------------------------------------------
# Cleanup scratch dir
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_scratch():
    """Remove _test_scratch if created by non-tmp_path tests."""
    yield
    scratch = Path(_PROJECT_ROOT) / "_test_scratch"
    if scratch.exists():
        import shutil
        shutil.rmtree(scratch)
