#!/usr/bin/env python3
"""Create and optionally enqueue single-trial queue manifests for dg_bidmc.

One manifest is generated per trial so the ray-hetzner queue daemon remains
the only scheduler. Each manifest packages the project code as a tar.gz
artifact, injects the trial config via METAOPT_EXPERIMENT_CONFIG_JSON, and
points the results_contract at metrics.json written by scripts/ray_runner.py.

Usage:
    python3 scripts/enqueue_trial_fanout.py \\
        --trials-file trials.json \\
        --iteration 5 \\
        --submission-name screen-round1 \\
        --queue-root /home/jakub/projects/dg_bidmc/.ml-metaopt \\
        [--enqueue] [--dry-run]

The entrypoint runs from the Aorus Ray venv against the unpacked workspace:
    /home/jakub/ray-venv/bin/python3 scripts/ray_runner.py

which reads METAOPT_EXPERIMENT_CONFIG_JSON and writes metrics.json to
METAOPT_WORKSPACE/results (both set by the queue backend before invocation).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN_ID = "dg-gat-respiratory-v2"
DEFAULT_ENTRYPOINT = "/home/jakub/ray-venv/bin/python3 scripts/ray_runner.py"
DEFAULT_METRICS_FILE = "metrics.json"
DEFAULT_QUEUE_HELPER = Path("/home/jakub/projects/ray-hetzner/metaopt/enqueue_batch.py")
DEFAULT_MANIFESTS_DIR = PROJECT_ROOT / ".ml-metaopt" / "artifacts" / "manifests"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / ".ml-metaopt" / "artifacts" / "code"
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9]+")
_METADATA_KEYS = {"name", "trial_name", "description", "experiment", "config"}
_EXCLUDED_PARTS = {
    ".git", ".ml-metaopt", "data", "results", "logs",
    "__pycache__", ".pytest_cache",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trials(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        trials = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            trials = payload
        elif isinstance(payload, dict) and isinstance(payload.get("trials"), list):
            trials = payload["trials"]
        else:
            raise ValueError(
                "Trials file must contain a JSON array, a {'trials': [...]} object, or JSONL lines"
            )
    if not all(isinstance(t, dict) for t in trials):
        raise ValueError("Every trial entry must be a JSON object")
    return trials


def deep_merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def sanitize_component(value: str) -> str:
    text = _SANITIZE_RE.sub("-", value.strip().lower()).strip("-")
    return text or "trial"


def build_batch_id(iteration: int, submission_name: str, trial_index: int, trial_name: str) -> str:
    return (
        f"iter{iteration}-"
        f"{sanitize_component(submission_name)}-"
        f"{trial_index:03d}-"
        f"{sanitize_component(trial_name)}"
    )


def should_exclude(relative_path: Path) -> bool:
    if any(part in _EXCLUDED_PARTS for part in relative_path.parts):
        return True
    if relative_path.name.endswith(".pyc"):
        return True
    return False


def package_code_artifact(project_root: Path, artifact_path: Path) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(artifact_path, "w:gz") as archive:
        for path in sorted(project_root.rglob("*")):
            relative = path.relative_to(project_root)
            if should_exclude(relative):
                continue
            archive.add(path, arcname=str(relative))
    return artifact_path


def build_manifest(
    *,
    campaign_id: str,
    iteration: int,
    batch_id: str,
    artifact_path: Path,
    entrypoint: str,
    trial_config: dict[str, Any],
    experiment: dict[str, Any],
    retry_max_attempts: int,
) -> dict[str, Any]:
    """Build a single-trial queue manifest satisfying the ray-hetzner backend contract."""
    return {
        "version": 1,
        "campaign_id": campaign_id,
        "iteration": iteration,
        "batch_id": batch_id,
        "experiment": experiment,
        "artifacts": {
            "code_artifact": {
                "uri": f"file://{artifact_path}",
            }
        },
        "execution": {
            "entrypoint": entrypoint,
            "env": {
                # Injected as env var; ray_runner.py reads this under the new contract.
                "METAOPT_EXPERIMENT_CONFIG_JSON": json.dumps(trial_config, sort_keys=True),
            },
        },
        "results_contract": {
            "metrics_file": DEFAULT_METRICS_FILE,
        },
        "retry_policy": {
            "max_attempts": retry_max_attempts,
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def enqueue_manifest(
    queue_helper: Path,
    manifest_path: Path,
    queue_root: str,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    cmd = [
        "python3",
        str(queue_helper),
        "--manifest",
        str(manifest_path),
        "--queue-root",
        queue_root,
    ]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return {"stdout": result.stdout.strip(), "dry_run": dry_run}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and optionally enqueue single-trial queue manifests."
    )
    parser.add_argument("--trials-file", required=True, help="JSON/JSONL file with trial configs.")
    parser.add_argument("--base-config", help="Optional JSON base config merged into every trial.")
    parser.add_argument(
        "--queue-root",
        default="/home/jakub/projects/dg_bidmc/.ml-metaopt",
        help="Explicit per-project remote queue root.",
    )
    parser.add_argument("--iteration", required=True, type=int, help="Iteration number.")
    parser.add_argument("--submission-name", required=True, help="Stable label for this fan-out submission.")
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--entrypoint", default=DEFAULT_ENTRYPOINT)
    parser.add_argument("--retry-max-attempts", type=int, default=2)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--manifests-dir", default=str(DEFAULT_MANIFESTS_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--queue-helper", default=str(DEFAULT_QUEUE_HELPER))
    parser.add_argument("--enqueue", action="store_true", help="Submit manifests through ray-hetzner.")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to enqueue_batch.py.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    manifests_root = Path(args.manifests_dir).resolve()
    artifacts_root = Path(args.artifacts_dir).resolve()
    queue_helper = Path(args.queue_helper).resolve()
    submission_name = sanitize_component(args.submission_name)
    submission_root = manifests_root / submission_name
    configs_root = submission_root / "configs"
    manifests_out = submission_root / "batches"
    artifact_path = artifacts_root / f"{submission_name}.tar.gz"

    base_config = load_json(Path(args.base_config)) if args.base_config else {}
    trials = load_trials(Path(args.trials_file))
    package_code_artifact(project_root, artifact_path)

    summary_trials: list[dict[str, Any]] = []
    for index, trial in enumerate(trials, start=1):
        metadata = dict(trial)
        override = metadata.pop("config", None)
        if override is None:
            override = {key: value for key, value in metadata.items() if key not in _METADATA_KEYS}
        trial_name = metadata.get("trial_name") or metadata.get("name") or f"trial-{index:03d}"
        full_config = deep_merge_configs(base_config, override)
        batch_id = build_batch_id(args.iteration, submission_name, index, trial_name)
        config_path = configs_root / f"{batch_id}.json"
        write_json(config_path, full_config)

        manifest = build_manifest(
            campaign_id=args.campaign_id,
            iteration=args.iteration,
            batch_id=batch_id,
            artifact_path=artifact_path,
            entrypoint=args.entrypoint,
            trial_config=full_config,
            experiment={
                "submission_name": submission_name,
                "trial_index": index,
                "trial_name": trial_name,
                "description": metadata.get("description"),
            },
            retry_max_attempts=args.retry_max_attempts,
        )
        manifest_path = manifests_out / f"{batch_id}.json"
        write_json(manifest_path, manifest)

        enqueue_result = None
        if args.enqueue:
            enqueue_result = enqueue_manifest(
                queue_helper, manifest_path, args.queue_root, dry_run=args.dry_run
            )

        summary_trials.append(
            {
                "batch_id": batch_id,
                "trial_name": trial_name,
                "manifest_path": str(manifest_path),
                "config_path": str(config_path),
                "enqueue_result": enqueue_result,
            }
        )

    submission_path = submission_root / "submission.json"
    submission = {
        "created_at": now_iso(),
        "campaign_id": args.campaign_id,
        "iteration": args.iteration,
        "submission_name": submission_name,
        "queue_root": args.queue_root,
        "artifact_path": str(artifact_path),
        "entrypoint": args.entrypoint,
        "enqueue": args.enqueue,
        "dry_run": args.dry_run,
        "trial_count": len(summary_trials),
        "trials": summary_trials,
    }
    write_json(submission_path, submission)
    print(json.dumps({"submission_path": str(submission_path), "trial_count": len(summary_trials)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
