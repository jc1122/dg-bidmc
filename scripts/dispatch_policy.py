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


def classify_stage(config: dict) -> str:
    """Return the dispatch stage for *config*, defaulting to 'train_gpu'."""
    return config.get("dispatch_stage", "train_gpu")


def default_screen_config(config: dict) -> dict:
    """Return a lightweight screening copy of *config*.

    Caps epochs/patience, forces single ensemble and coarse postprocess
    while preserving model architecture keys.
    """
    out = copy.deepcopy(config)
    out["dispatch_stage"] = "screen_cpu"
    if "max_epochs" in out:
        out["max_epochs"] = min(out["max_epochs"], 25)
    if "patience" in out:
        out["patience"] = min(out["patience"], 5)
    out["n_ensemble"] = 1
    out["postprocess_search"] = "coarse"
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
