"""Tests for stage-aware dispatch_trial (Task 3).

All tests are offline — no live Ray cluster required.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dispatch_policy import prepare_stage_config  # noqa: E402

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# No-mutation guarantee for the dispatch live path
# ---------------------------------------------------------------------------


class TestDispatchNoMutation:
    """Verify the config-prep pattern used by dispatch() never mutates input."""

    def test_resolve_stage_does_not_mutate(self):
        from dispatch_trial import _resolve_stage
        cfg = {"dispatch_stage": "train_gpu", "lr": 0.01}
        snapshot = copy.deepcopy(cfg)
        _resolve_stage(cfg, cpu_only=False)
        assert cfg == snapshot

    def test_resolve_stage_cpu_only_does_not_mutate(self):
        from dispatch_trial import _resolve_stage
        cfg = {"dispatch_stage": "train_gpu", "lr": 0.01}
        snapshot = copy.deepcopy(cfg)
        _resolve_stage(cfg, cpu_only=True)
        assert cfg == snapshot

    def test_full_prep_path_does_not_mutate(self):
        """Simulate the exact code path dispatch() uses before serializing."""
        from dispatch_trial import _resolve_stage

        config = {
            "dispatch_stage": "screen_cpu",
            "max_epochs": 200,
            "patience": 30,
            "nested": {"a": 1},
        }
        snapshot = copy.deepcopy(config)

        stage = _resolve_stage(config, cpu_only=False)
        shaped = prepare_stage_config(config, stage)

        assert config == snapshot, "dispatch config-prep mutated the caller dict"
        assert shaped is not config
        assert shaped["max_epochs"] == 25
        assert shaped["_screening"] is True

    def test_dry_run_does_not_mutate_caller_config(self, tmp_path):
        """End-to-end: dry-run must leave the caller's config untouched."""
        from dispatch_trial import dispatch

        config = {"dispatch_stage": "screen_cpu", "max_epochs": 200}
        snapshot = copy.deepcopy(config)

        out = tmp_path / "manifest.json"
        dispatch(config, Path(_PROJECT_ROOT), out, dry_run=True)

        assert config == snapshot, "dry-run mutated the caller config"


# ---------------------------------------------------------------------------
# Canonical shaping consistency
# ---------------------------------------------------------------------------


class TestCanonicalShaping:
    """Verify dry-run and live paths use the same canonical shaping."""

    def test_dry_run_manifest_uses_canonical_shaping(self, tmp_path):
        """The shaped config in a dry-run manifest must match
        prepare_stage_config applied directly."""
        from dispatch_trial import dispatch

        config = {
            "dispatch_stage": "screen_cpu",
            "max_epochs": 200,
            "patience": 30,
        }
        out = tmp_path / "manifest.json"
        manifest = dispatch(config, Path(_PROJECT_ROOT), out, dry_run=True)

        expected = prepare_stage_config(config, "screen_cpu")
        assert manifest["config"] == expected

    def test_screen_stage_includes_postprocess_coarse(self, tmp_path):
        """Screen-stage shaping must set postprocess_search=coarse."""
        from dispatch_trial import dispatch

        config = {"dispatch_stage": "screen_cpu"}
        out = tmp_path / "manifest.json"
        manifest = dispatch(config, Path(_PROJECT_ROOT), out, dry_run=True)

        assert manifest["config"]["postprocess_search"] == "coarse"
        assert manifest["config"]["_screening"] is True


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
