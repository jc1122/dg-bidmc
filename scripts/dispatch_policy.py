"""Staged dispatch policy helpers for two-tier CPU/GPU trial flow.

Pure functions — no Ray, no network, no side effects.
Compatible with delegated ray-hetzner queue execution.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass
class DispatchPolicy:
    """Resource budget knobs for staged dispatch."""

    cpu_screen_max_concurrent: int = 4
    cpu_screen_cpus_per_trial: int = 2
    gpu_promotion_max_concurrent: int = 2
    gpu_promotion_gpus_per_trial: int = 1
    cpu_eval_max_concurrent: int = 4


# Caps applied during screen_cpu stage to reduce cost
_SCREEN_MAX_EPOCHS = 25
_SCREEN_MAX_PATIENCE = 5


def classify_stage(config: dict) -> str:
    """Return the dispatch stage for *config*, defaulting to 'train_gpu'."""
    return config.get("dispatch_stage", "train_gpu")


def prepare_stage_config(config: dict, stage: str | None = None) -> dict:
    """Return a config shaped for *stage*, without mutating the original.

    This is the **canonical** shaping function used by all dispatch paths
    (dry-run, live, standalone runner).

    Stages:
        screen_cpu  -- cap epochs/patience, force n_ensemble=1, coarse
                       postprocess, mark ``_screening``
        train_gpu   -- full training, no caps
        eval_cpu    -- CPU-lane evaluation, mark ``_eval_only``

    If *stage* is ``None``, it is read from ``config["dispatch_stage"]``
    (defaulting to ``"train_gpu"``).
    """
    out = copy.deepcopy(config)
    if stage is None:
        stage = out.get("dispatch_stage", "train_gpu")

    out["dispatch_stage"] = stage

    if stage == "screen_cpu":
        out["max_epochs"] = min(
            out.get("max_epochs", _SCREEN_MAX_EPOCHS), _SCREEN_MAX_EPOCHS,
        )
        out["patience"] = min(
            out.get("patience", _SCREEN_MAX_PATIENCE), _SCREEN_MAX_PATIENCE,
        )
        out["n_ensemble"] = 1
        out["postprocess_search"] = "coarse"
        out["_screening"] = True
    elif stage == "eval_cpu":
        out["_eval_only"] = True
    # train_gpu: no caps — preserve full training behaviour

    return out


def resources_for_stage(stage: str, policy: DispatchPolicy) -> dict[str, int]:
    """Map *stage* to generic Ray resource requirements."""
    if stage == "screen_cpu":
        return {"num_cpus": policy.cpu_screen_cpus_per_trial}
    if stage == "eval_cpu":
        return {"num_cpus": 1}
    if stage == "train_gpu":
        return {"num_gpus": policy.gpu_promotion_gpus_per_trial}
    raise ValueError(f"Unknown stage: {stage!r}")


def promotion_candidates(
    screen_results: list[dict],
    min_score: float,
    top_k: int,
) -> list[dict]:
    """Filter and rank screening results for GPU promotion.

    Keeps entries with ``boundary_f1_600ms >= min_score``, sorts descending,
    and returns at most *top_k* entries.
    """
    qualified = [r for r in screen_results if r.get("boundary_f1_600ms", 0) >= min_score]
    qualified.sort(key=lambda r: r.get("boundary_f1_600ms", 0), reverse=True)
    return qualified[:top_k]
