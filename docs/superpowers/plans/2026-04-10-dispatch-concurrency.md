# Dispatch Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current single-trial, named-host dispatch setup with a project-local, delegated two-tier dispatch flow that uses generic Ray resources, parallel CPU screening, and bounded GPU promotion.

**Architecture:** Add a small pure-Python dispatch policy layer that classifies jobs, computes generic resource requests, and decides promotion. Keep `scripts/ray_runner.py` as the per-trial executor, refactor `scripts/dispatch_trial.py` into a generic stage-aware launcher, and add a local manifest builder so delegated Hetzner runs can fan out multiple CPU screening trials without encoding machine identity.

**Tech Stack:** Python 3, Ray, JSON manifests, pytest, existing metaopt campaign YAML

---

## File structure

- **Create:** `scripts/dispatch_policy.py`  
  Pure helpers for stage classification, concurrency policy parsing, resource selection, and GPU-promotion decisions.

- **Create:** `scripts/build_batch_manifest.py`  
  Local helper that turns a list of configs plus policy into a delegated batch manifest with multiple generic-resource trials.

- **Modify:** `scripts/dispatch_trial.py`  
  Remove named-resource assumptions, use generic Ray resources, add stage-aware `.options(...)`, and add a dry-run manifest path for validation.

- **Modify:** `scripts/ray_runner.py`  
  Add stage-aware config shaping for CPU screening vs GPU promotion and record stage provenance in results.

- **Modify:** `ml_metaopt_campaign.yaml`  
  Add project-local concurrency knobs for CPU screening, GPU promotion, and evaluation lanes while keeping remote execution delegated.

- **Modify:** `README.md`  
  Document the new dispatch model and the fact that this repo emits generic resource requirements only.

- **Modify:** `AGENTS.md`  
  Update operator guidance for strict CPU-screen then GPU-promotion dispatch and delegated Hetzner execution.

- **Test:** `tests/test_dispatch_policy.py`  
  Unit tests for policy parsing, stage classification, resource maps, and promotion rules.

- **Test:** `tests/test_batch_manifest.py`  
  Snapshot-style tests for generated manifests proving absence of named resource labels.

---

### Task 1: Add pure dispatch policy helpers

**Files:**
- Create: `scripts/dispatch_policy.py`
- Test: `tests/test_dispatch_policy.py`

- [ ] **Step 1: Write the failing policy tests**

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dispatch_policy import (
    DispatchPolicy,
    classify_stage,
    default_screen_config,
    promotion_candidates,
    resources_for_stage,
)


def test_classify_stage_uses_explicit_stage_first():
    assert classify_stage({"dispatch_stage": "screen_cpu"}) == "screen_cpu"
    assert classify_stage({"dispatch_stage": "train_gpu"}) == "train_gpu"


def test_default_screen_config_reduces_cost_but_preserves_model_keys():
    screen = default_screen_config(
        {
            "arch": "transformer",
            "hidden_dim": 64,
            "max_epochs": 100,
            "patience": 15,
            "n_ensemble": 1,
        }
    )
    assert screen["arch"] == "transformer"
    assert screen["max_epochs"] == 25
    assert screen["patience"] == 5
    assert screen["dispatch_stage"] == "screen_cpu"


def test_resources_for_stage_never_returns_named_host_labels():
    policy = DispatchPolicy(
        cpu_screen_max_concurrent=6,
        cpu_screen_cpus_per_trial=4,
        gpu_promotion_max_concurrent=1,
        gpu_promotion_gpus_per_trial=1,
        cpu_eval_max_concurrent=4,
    )
    assert resources_for_stage("screen_cpu", policy) == {"num_cpus": 4}
    assert resources_for_stage("train_gpu", policy) == {"num_gpus": 1}


def test_promotion_candidates_uses_threshold_then_top_k():
    screen_results = [
        {"trial_id": "t1", "boundary_f1_600ms": 0.83},
        {"trial_id": "t2", "boundary_f1_600ms": 0.91},
        {"trial_id": "t3", "boundary_f1_600ms": 0.89},
    ]
    promoted = promotion_candidates(screen_results, min_score=0.88, top_k=2)
    assert [item["trial_id"] for item in promoted] == ["t2", "t3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dispatch_policy.py -q`  
Expected: `ImportError` / missing `dispatch_policy`

- [ ] **Step 3: Write the minimal helper module**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DispatchPolicy:
    cpu_screen_max_concurrent: int
    cpu_screen_cpus_per_trial: int
    gpu_promotion_max_concurrent: int
    gpu_promotion_gpus_per_trial: int
    cpu_eval_max_concurrent: int


def classify_stage(config: dict) -> str:
    return config.get("dispatch_stage", "train_gpu")


def default_screen_config(config: dict) -> dict:
    screen = dict(config)
    screen["dispatch_stage"] = "screen_cpu"
    screen["max_epochs"] = min(int(screen.get("max_epochs", 100)), 25)
    screen["patience"] = min(int(screen.get("patience", 15)), 5)
    screen["n_ensemble"] = 1
    screen["postprocess_search"] = "coarse"
    return screen


def resources_for_stage(stage: str, policy: DispatchPolicy) -> dict[str, int]:
    if stage == "screen_cpu":
        return {"num_cpus": policy.cpu_screen_cpus_per_trial}
    if stage == "eval_cpu":
        return {"num_cpus": 1}
    if stage == "train_gpu":
        return {"num_gpus": policy.gpu_promotion_gpus_per_trial}
    raise ValueError(f"Unknown dispatch stage: {stage}")


def promotion_candidates(screen_results: list[dict], min_score: float, top_k: int) -> list[dict]:
    ranked = sorted(
        [row for row in screen_results if row.get("boundary_f1_600ms", 0.0) >= min_score],
        key=lambda row: row.get("boundary_f1_600ms", 0.0),
        reverse=True,
    )
    return ranked[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dispatch_policy.py -q`  
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/dispatch_policy.py tests/test_dispatch_policy.py
git commit -m "feat: add dispatch policy helpers"
```

### Task 2: Add local batch-manifest generation for delegated runs

**Files:**
- Create: `scripts/build_batch_manifest.py`
- Test: `tests/test_batch_manifest.py`

- [ ] **Step 1: Write the failing manifest tests**

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_batch_manifest import build_batch_manifest
from dispatch_policy import DispatchPolicy


def test_build_batch_manifest_emits_multiple_screen_trials():
    policy = DispatchPolicy(6, 4, 1, 1, 4)
    manifest = build_batch_manifest(
        campaign_id="dg-gat-respiratory-v2",
        iteration=16,
        trial_configs=[
            {"trial_id": "screen-1", "dispatch_stage": "screen_cpu", "arch": "gat"},
            {"trial_id": "screen-2", "dispatch_stage": "screen_cpu", "arch": "transformer"},
        ],
        policy=policy,
    )
    assert len(manifest["trials"]) == 2
    assert manifest["trials"][0]["resources"] == {"num_cpus": 4}


def test_build_batch_manifest_gpu_trial_has_no_named_resource():
    policy = DispatchPolicy(6, 4, 1, 1, 4)
    manifest = build_batch_manifest(
        campaign_id="dg-gat-respiratory-v2",
        iteration=16,
        trial_configs=[{"trial_id": "gpu-1", "dispatch_stage": "train_gpu", "arch": "transformer"}],
        policy=policy,
    )
    resources = manifest["trials"][0]["resources"]
    assert resources == {"num_gpus": 1}
    assert "aorus" not in resources
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_batch_manifest.py -q`  
Expected: `ImportError` / missing `build_batch_manifest`

- [ ] **Step 3: Write the manifest builder**

```python
from dispatch_policy import classify_stage, resources_for_stage


def build_batch_manifest(campaign_id: str, iteration: int, trial_configs: list[dict], policy, entrypoint: str = "python3 /root/dg_bidmc/scripts/ray_runner.py") -> dict:
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


if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    from dispatch_policy import DispatchPolicy

    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--trial-configs", required=True, help="Path to a JSON list of config dicts")
    args = parser.parse_args()

    trial_configs = json.loads(Path(args.trial_configs).read_text())
    policy = DispatchPolicy(6, 4, 1, 1, 4)
    manifest = build_batch_manifest(args.campaign_id, args.iteration, trial_configs, policy)
    print(json.dumps(manifest, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_batch_manifest.py -q`  
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_batch_manifest.py tests/test_batch_manifest.py
git commit -m "feat: add delegated batch manifest builder"
```

### Task 3: Refactor the Ray launcher to use generic staged resources

**Files:**
- Modify: `scripts/dispatch_trial.py`
- Modify: `scripts/ray_runner.py`
- Test: `tests/test_dispatch_policy.py`
- Test: `tests/test_batch_manifest.py`

- [ ] **Step 1: Write the failing dry-run launcher test**

```python
from pathlib import Path
import subprocess


def test_dispatch_trial_dry_run_prints_generic_gpu_manifest(tmp_path):
    out = tmp_path / "dry-run.json"
    cmd = [
        "python3",
        "scripts/dispatch_trial.py",
        "--config",
        '{"trial_id":"gpu-1","dispatch_stage":"train_gpu","arch":"transformer"}',
        "--out",
        str(out),
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert '"num_gpus": 1' in result.stdout
    assert '"aorus"' not in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dispatch_policy.py tests/test_batch_manifest.py -q`  
Expected: failure because `--dry-run` path does not exist yet

- [ ] **Step 3: Refactor `scripts/dispatch_trial.py`**

```python
from build_batch_manifest import build_batch_manifest
from dispatch_policy import DispatchPolicy, classify_stage, resources_for_stage


@ray.remote
def run_on_cpu(config_json: str) -> str:
    import json, os, sys, traceback
    cwd = os.getcwd()
    sys.path.insert(0, os.path.join(cwd, "scripts"))
    os.environ["DG_PROJECT_ROOT"] = cwd
    try:
        from ray_runner import run_trial, get_default_config
        full = get_default_config()
        full.update(json.loads(config_json))
        full["dispatch_stage"] = "screen_cpu"
        return json.dumps(run_trial(full))
    except Exception as exc:
        return json.dumps({"error": str(exc), "traceback": traceback.format_exc(), "status": "FAILED"})


@ray.remote(num_gpus=1)
def run_on_gpu(config_json: str) -> str:
    import json, os, sys, traceback
    cwd = os.getcwd()
    sys.path.insert(0, os.path.join(cwd, "scripts"))
    os.environ["DG_PROJECT_ROOT"] = cwd
    try:
        from ray_runner import run_trial, get_default_config
        full = get_default_config()
        full.update(json.loads(config_json))
        full["dispatch_stage"] = "train_gpu"
        return json.dumps(run_trial(full))
    except Exception as exc:
        return json.dumps({"error": str(exc), "traceback": traceback.format_exc(), "status": "FAILED"})


def dispatch(config: dict, project_root: Path, output_path: Path, cpu_only: bool = False, gpu_retries: int = 2, dry_run: bool = False):
    policy = DispatchPolicy(6, 4, 1, 1, 4)
    stage = "screen_cpu" if cpu_only else classify_stage(config)
    manifest = build_batch_manifest(
        campaign_id="local-dispatch",
        iteration=0,
        trial_configs=[{**config, "dispatch_stage": stage, "trial_id": config.get("trial_id", "trial-1")}],
        policy=policy,
        entrypoint="python3 /root/dg_bidmc/scripts/ray_runner.py",
    )
    if dry_run:
        print(json.dumps(manifest, indent=2))
        output_path.write_text(json.dumps(manifest, indent=2))
        return manifest

    resource_opts = resources_for_stage(stage, policy)
    hide_gitignore(project_root)
    try:
        runtime = make_runtime_env(project_root)
        ray.init(address="auto", runtime_env=runtime)
    finally:
        restore_gitignore(project_root)
    if stage == "screen_cpu":
        ref = run_on_cpu.options(**resource_opts).remote(json.dumps({**config, "dispatch_stage": stage}))
    elif stage == "train_gpu":
        ref = run_on_gpu.options(**resource_opts).remote(json.dumps({**config, "dispatch_stage": stage}))
    else:
        ref = run_on_cpu.options(**resource_opts).remote(json.dumps({**config, "dispatch_stage": "eval_cpu"}))
    result_json = ray.get(ref)
    result = json.loads(result_json)
    output_path.write_text(json.dumps(result, indent=2))
    ray.shutdown()
    return result
```

- [ ] **Step 4: Add stage-aware config shaping in `scripts/ray_runner.py`**

```python
def prepare_stage_config(config: dict) -> dict:
    full = dict(config)
    stage = full.get("dispatch_stage", "train_gpu")
    if stage == "screen_cpu":
        full["max_epochs"] = min(int(full.get("max_epochs", 100)), 25)
        full["patience"] = min(int(full.get("patience", 15)), 5)
        full["n_ensemble"] = 1
        full["screening_mode"] = True
    return full


def main():
    config_json = os.environ.get("METAOPT_TRIAL_CONFIG", "{}")
    result_path = os.environ.get("METAOPT_RESULT_PATH", "result.json")
    trial_config = json.loads(config_json)
    config = get_default_config()
    config.update(trial_config)
    config = prepare_stage_config(config)
    result = run_trial(config)
    Path(result_path).write_text(json.dumps(result, indent=2))
```

- [ ] **Step 5: Run the targeted tests**

Run: `python3 -m pytest tests/test_dispatch_policy.py tests/test_batch_manifest.py -q`  
Expected: all targeted dispatch tests pass

- [ ] **Step 6: Run launcher dry runs**

Run:

```bash
python3 scripts/dispatch_trial.py --config '{"trial_id":"screen-1","dispatch_stage":"screen_cpu","arch":"gat"}' --out /tmp/screen.json --dry-run
python3 scripts/dispatch_trial.py --config '{"trial_id":"gpu-1","dispatch_stage":"train_gpu","arch":"transformer"}' --out /tmp/gpu.json --dry-run
```

Expected:
- `/tmp/screen.json` contains `"resources": {"num_cpus": 4}`
- `/tmp/gpu.json` contains `"resources": {"num_gpus": 1}`
- neither file contains `"aorus"`

- [ ] **Step 7: Commit**

```bash
git add scripts/dispatch_trial.py scripts/ray_runner.py
git commit -m "feat: make dispatch stage-aware and generic-resource"
```

### Task 4: Update campaign config and operator docs

**Files:**
- Modify: `ml_metaopt_campaign.yaml`
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Write the failing config/documentation checks**

```python
from pathlib import Path


def test_campaign_yaml_mentions_cpu_screen_and_gpu_promotion():
    text = Path("ml_metaopt_campaign.yaml").read_text()
    assert "cpu_screen_max_concurrent" in text
    assert "gpu_promotion_max_concurrent" in text
    assert '"aorus": 1' not in text
```

- [ ] **Step 2: Run check to verify current failure**

Run: `python3 -m pytest tests/test_dispatch_policy.py -q`  
Expected: failure until campaign keys and docs are updated

- [ ] **Step 3: Update the campaign config**

```yaml
dispatch_policy:
  background_slots: 4
  auxiliary_slots: 2
  cpu_screen_max_concurrent: 6
  cpu_screen_cpus_per_trial: 4
  gpu_promotion_max_concurrent: 1
  gpu_promotion_gpus_per_trial: 1
  cpu_eval_max_concurrent: 4
  promotion_min_score: 0.88
  promotion_top_k: 1

execution:
  runner_type: ray_queue_runner
  entrypoint: python3 /srv/metaopt/project/scripts/ray_runner.py
  target_cluster_utilization: 0.90
  trial_budget:
    kind: staged_trials
    screen_cpu: 6
    train_gpu: 1
```

- [ ] **Step 4: Update operator docs**

```md
## Dispatch model

- CPU screening jobs use generic CPU Ray resources only
- GPU promotion jobs use `num_gpus: 1` only
- This repository does not emit named-host resource labels
- Remote execution remains delegated through the Hetzner workflow
```

- [ ] **Step 5: Run targeted tests and smoke docs check**

Run:

```bash
python3 -m pytest tests/test_dispatch_policy.py tests/test_batch_manifest.py -q
rg '"aorus": 1|resources=\\{\"aorus\"' . -n
```

Expected:
- dispatch tests pass
- no remaining project dispatch path emits named-host resource assumptions

- [ ] **Step 6: Commit**

```bash
git add ml_metaopt_campaign.yaml README.md AGENTS.md
git commit -m "docs: configure staged generic-resource dispatch"
```

### Task 5: Final verification and delegated handoff

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-dispatch-concurrency-design.md` only if implementation deviates

- [ ] **Step 1: Run the local verification set**

Run:

```bash
python3 -m pytest tests/test_dispatch_policy.py tests/test_batch_manifest.py -q
python3 scripts/dispatch_trial.py --config '{"trial_id":"screen-1","dispatch_stage":"screen_cpu","arch":"gat"}' --out /tmp/screen.json --dry-run
python3 scripts/dispatch_trial.py --config '{"trial_id":"gpu-1","dispatch_stage":"train_gpu","arch":"transformer"}' --out /tmp/gpu.json --dry-run
python3 - <<'PY'
import json
for path in ["/tmp/screen.json", "/tmp/gpu.json"]:
    with open(path) as f:
        data = json.load(f)
    print(path, data["trials"][0]["resources"])
PY
```

Expected:
- tests pass
- CPU manifest prints `{'num_cpus': 4}`
- GPU manifest prints `{'num_gpus': 1}`

- [ ] **Step 2: Prepare delegated execution handoff**

```bash
python3 scripts/build_batch_manifest.py > /tmp/dg_bidmc_batch_manifest.json
python3 - <<'PY'
import json
with open('/tmp/dg_bidmc_batch_manifest.json') as f:
    data = json.load(f)
assert all('aorus' not in trial['resources'] for trial in data['trials'])
print('generic-resource manifest ready')
PY
```

- [ ] **Step 3: Commit**

```bash
git add scripts/build_batch_manifest.py scripts/dispatch_trial.py scripts/ray_runner.py tests/test_dispatch_policy.py tests/test_batch_manifest.py ml_metaopt_campaign.yaml README.md AGENTS.md
git commit -m "feat: enable staged delegated dispatch concurrency"
```

## Self-review

- **Spec coverage:** covered the local-only constraint, generic resource requests, CPU-screen vs GPU-promotion lanes, concurrency caps, manifest changes, and delegated execution boundary.
- **Placeholder scan:** no `TBD` / `TODO` placeholders remain in the tasks.
- **Type consistency:** uses the same stage names throughout: `screen_cpu`, `train_gpu`, `eval_cpu`.
