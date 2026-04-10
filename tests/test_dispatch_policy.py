"""Tests for scripts.dispatch_policy — staged dispatch helpers."""

import copy

import pytest
from scripts.dispatch_policy import (
    DispatchPolicy,
    classify_stage,
    prepare_stage_config,
    promotion_candidates,
    resources_for_stage,
)


# ---------------------------------------------------------------------------
# classify_stage
# ---------------------------------------------------------------------------

class TestClassifyStage:
    def test_returns_dispatch_stage_when_present(self):
        assert classify_stage({"dispatch_stage": "screen_cpu"}) == "screen_cpu"

    def test_defaults_to_train_gpu(self):
        assert classify_stage({"lr": 0.01}) == "train_gpu"

    def test_empty_config(self):
        assert classify_stage({}) == "train_gpu"


# ---------------------------------------------------------------------------
# prepare_stage_config (canonical shaping function)
# ---------------------------------------------------------------------------

class TestPrepareStageConfig:
    """Covers all stages, caps, defaults, flags, and no-mutation guarantees."""

    # -- screen_cpu --------------------------------------------------------

    def test_screen_cpu_caps_max_epochs(self):
        cfg = prepare_stage_config({"max_epochs": 100}, "screen_cpu")
        assert cfg["max_epochs"] == 25

    def test_screen_cpu_keeps_low_epochs(self):
        cfg = prepare_stage_config({"max_epochs": 10}, "screen_cpu")
        assert cfg["max_epochs"] == 10

    def test_screen_cpu_defaults_missing_max_epochs(self):
        cfg = prepare_stage_config({}, "screen_cpu")
        assert cfg["max_epochs"] == 25

    def test_screen_cpu_caps_patience(self):
        cfg = prepare_stage_config({"patience": 20}, "screen_cpu")
        assert cfg["patience"] == 5

    def test_screen_cpu_keeps_low_patience(self):
        cfg = prepare_stage_config({"patience": 3}, "screen_cpu")
        assert cfg["patience"] == 3

    def test_screen_cpu_defaults_missing_patience(self):
        cfg = prepare_stage_config({}, "screen_cpu")
        assert cfg["patience"] == 5

    def test_screen_cpu_forces_ensemble_1(self):
        cfg = prepare_stage_config({"n_ensemble": 5}, "screen_cpu")
        assert cfg["n_ensemble"] == 1

    def test_screen_cpu_sets_coarse_postprocess(self):
        cfg = prepare_stage_config({"postprocess_search": "fine"}, "screen_cpu")
        assert cfg["postprocess_search"] == "coarse"

    def test_screen_cpu_marks_screening(self):
        cfg = prepare_stage_config({}, "screen_cpu")
        assert cfg["_screening"] is True
        assert cfg["dispatch_stage"] == "screen_cpu"

    def test_screen_cpu_preserves_model_keys(self):
        cfg = {"arch": "GAT", "hidden": 64, "heads": 8}
        out = prepare_stage_config(cfg, "screen_cpu")
        assert out["arch"] == "GAT"
        assert out["hidden"] == 64
        assert out["heads"] == 8

    # -- train_gpu ---------------------------------------------------------

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

    # -- eval_cpu ----------------------------------------------------------

    def test_eval_cpu_marks_eval_only(self):
        cfg = prepare_stage_config({}, "eval_cpu")
        assert cfg["_eval_only"] is True
        assert cfg["dispatch_stage"] == "eval_cpu"

    # -- stage inference ---------------------------------------------------

    def test_stage_from_config_when_none(self):
        cfg = prepare_stage_config({"dispatch_stage": "eval_cpu"})
        assert cfg["_eval_only"] is True

    def test_default_stage_is_train_gpu(self):
        cfg = prepare_stage_config({"lr": 0.01})
        assert cfg["dispatch_stage"] == "train_gpu"

    # -- no-mutation guarantees --------------------------------------------

    def test_does_not_mutate_original(self):
        orig = {"max_epochs": 100, "patience": 20, "nested": {"a": 1}}
        snapshot = copy.deepcopy(orig)
        prepare_stage_config(orig, "screen_cpu")
        assert orig == snapshot

    def test_deep_copy_isolates_nested(self):
        inner = {"a": 1}
        orig = {"nested": inner}
        out = prepare_stage_config(orig, "screen_cpu")
        out["nested"]["a"] = 999
        assert inner["a"] == 1

    # -- idempotency -------------------------------------------------------

    def test_double_application_is_idempotent(self):
        cfg = {"max_epochs": 200, "patience": 30}
        once = prepare_stage_config(cfg, "screen_cpu")
        twice = prepare_stage_config(once, "screen_cpu")
        assert once == twice


# ---------------------------------------------------------------------------
# resources_for_stage
# ---------------------------------------------------------------------------

class TestResourcesForStage:
    def test_screen_cpu(self):
        p = DispatchPolicy()
        res = resources_for_stage("screen_cpu", p)
        assert res == {"num_cpus": p.cpu_screen_cpus_per_trial}
        assert "num_gpus" not in res

    def test_eval_cpu(self):
        res = resources_for_stage("eval_cpu", DispatchPolicy())
        assert res == {"num_cpus": 1}

    def test_train_gpu(self):
        p = DispatchPolicy(gpu_promotion_gpus_per_trial=2)
        res = resources_for_stage("train_gpu", p)
        assert res == {"num_gpus": 2}

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            resources_for_stage("magic", DispatchPolicy())

    def test_custom_policy_values(self):
        p = DispatchPolicy(cpu_screen_cpus_per_trial=4)
        assert resources_for_stage("screen_cpu", p) == {"num_cpus": 4}


# ---------------------------------------------------------------------------
# promotion_candidates
# ---------------------------------------------------------------------------

class TestPromotionCandidates:
    @pytest.fixture()
    def results(self):
        return [
            {"config": {"lr": 0.01}, "boundary_f1_600ms": 0.80},
            {"config": {"lr": 0.05}, "boundary_f1_600ms": 0.30},
            {"config": {"lr": 0.02}, "boundary_f1_600ms": 0.65},
            {"config": {"lr": 0.03}, "boundary_f1_600ms": 0.70},
        ]

    def test_filters_by_min_score(self, results):
        out = promotion_candidates(results, min_score=0.60, top_k=10)
        scores = [r["boundary_f1_600ms"] for r in out]
        assert all(s >= 0.60 for s in scores)
        assert len(out) == 3

    def test_sorted_descending(self, results):
        out = promotion_candidates(results, min_score=0.0, top_k=10)
        scores = [r["boundary_f1_600ms"] for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits(self, results):
        out = promotion_candidates(results, min_score=0.0, top_k=2)
        assert len(out) == 2
        assert out[0]["boundary_f1_600ms"] == 0.80
        assert out[1]["boundary_f1_600ms"] == 0.70

    def test_empty_input(self):
        assert promotion_candidates([], min_score=0.5, top_k=5) == []

    def test_none_qualify(self, results):
        assert promotion_candidates(results, min_score=0.99, top_k=5) == []

    def test_missing_key_treated_as_zero(self):
        """Entries lacking boundary_f1_600ms default to 0 in both filter and sort."""
        data = [
            {"config": {"lr": 0.01}, "boundary_f1_600ms": 0.50},
            {"config": {"lr": 0.02}},  # missing key
        ]
        # min_score > 0 filters out the missing-key entry
        out = promotion_candidates(data, min_score=0.1, top_k=10)
        assert len(out) == 1
        assert out[0]["boundary_f1_600ms"] == 0.50

    def test_missing_key_admitted_at_zero_threshold(self):
        """With min_score=0, missing-key entries pass filter and sort without KeyError."""
        data = [
            {"config": {"lr": 0.01}, "boundary_f1_600ms": 0.50},
            {"config": {"lr": 0.02}},  # missing key, defaults to 0
        ]
        out = promotion_candidates(data, min_score=0, top_k=10)
        assert len(out) == 2
        assert out[0]["boundary_f1_600ms"] == 0.50
        assert "boundary_f1_600ms" not in out[1]


# ---------------------------------------------------------------------------
# DispatchPolicy defaults
# ---------------------------------------------------------------------------

class TestDispatchPolicy:
    def test_default_values(self):
        p = DispatchPolicy()
        assert p.cpu_screen_max_concurrent >= 1
        assert p.cpu_screen_cpus_per_trial >= 1
        assert p.gpu_promotion_max_concurrent >= 1
        assert p.gpu_promotion_gpus_per_trial >= 1
        assert p.cpu_eval_max_concurrent >= 1

    def test_custom_values(self):
        p = DispatchPolicy(cpu_screen_max_concurrent=8, gpu_promotion_gpus_per_trial=2)
        assert p.cpu_screen_max_concurrent == 8
        assert p.gpu_promotion_gpus_per_trial == 2
