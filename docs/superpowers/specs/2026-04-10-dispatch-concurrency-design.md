# Dispatch concurrency design

## Problem

The project currently behaves like a single-lane remote optimizer:

- metaopt orchestration exposes `background_slots` and `auxiliary_slots`, but those are control-plane slots, not training concurrency
- the campaign uses `trial_budget: 1`, so only one remote training trial is issued per iteration
- the recorded remote manifest requests `{"num_gpus": 1, "aorus": 1}`, which assumes a named GPU host and does not fit a shared autoscaled Ray cluster

This is too conservative for a cluster that may include multiple Hetzner cx53 nodes plus one shared GPU host, and it is unfriendly to running three projects concurrently on the same Ray deployment.

## Goal

Rework this project’s dispatch so it:

1. uses more cluster resources concurrently
2. does not depend on named workers or named host resources
3. coexists fairly with other projects on the same Ray cluster
4. uses CPU capacity broadly for screening and reserves GPU for promoted candidates

## Recommended approach

Adopt a strict two-tier dispatch model:

1. **CPU screening lane** for broad parallel search on generic Ray CPU capacity
2. **GPU promotion lane** for a small number of higher-value training runs

This is preferred over GPU-first dispatch because the cluster has abundant CPU capacity relative to GPU capacity, and because a shared single-GPU environment should not be monopolized by one repo’s early-stage search.

## Alternatives considered

### 1. GPU-first opportunistic dispatch

Submit all trainable jobs as GPU-capable and let Ray place them opportunistically.

**Rejected** because it still bottlenecks on the shared GPU and increases interference risk with the other projects.

### 2. Mixed fixed lanes without strict promotion

Always run one GPU job and some CPU side jobs, but allow ad hoc bypass to the GPU lane.

**Rejected** because the promotion boundary becomes fuzzy and GPU usage becomes harder to control under shared-cluster contention.

### 3. Strict CPU screen then GPU promotion

Run many cheap CPU trials, promote only threshold-passing configs to a generic GPU lane.

**Accepted** because it gives the best CPU utilization while keeping GPU consumption deliberate and fair.

## Architecture

### Tier 1: CPU screening lane

CPU screen jobs must:

- request generic CPU resources only
- avoid named host resources
- be cheaper than full training runs
- be safe to run in parallel across multiple autoscaled cx53 nodes

Examples of screen-job reductions:

- shorter `max_epochs`
- reduced seeds
- smaller post-processing search
- no expensive ensemble or calibration passes

The purpose of this lane is ranking, not final reporting.

### Tier 2: GPU promotion lane

GPU promotion jobs must:

- request generic GPU resources only, such as `num_gpus: 1`
- avoid any `aorus`-style or host-specific resource label
- preserve current ROCm robustness behavior: retry, timeout, and CPU fallback where appropriate

The purpose of this lane is confirmatory training and reporting-quality evaluation for promoted configs.

## Resource model

### Generic placement only

This project must not encode host identity into manifests or dispatcher defaults. It should describe only resource needs:

- CPU screen jobs: CPU-only
- GPU promotion jobs: 1 GPU
- evaluation/post-processing jobs: CPU-only

Ray and the autoscaler choose placement.

### Fair-share controls

The project needs explicit concurrency caps so it can coexist with the other two projects.

Add config knobs for:

- max concurrent CPU screening jobs
- per-screen-job CPU request
- max concurrent GPU promotion jobs
- maximum share of the project’s own queue that can be in-flight

These caps should be local to this repo and not assume exclusive cluster ownership.

## Config changes

The campaign config should gain a dispatch block that distinguishes job classes:

```yaml
dispatch_policy:
  background_slots: 4
  auxiliary_slots: 2
  cpu_screen_max_concurrent: 6
  cpu_screen_cpus_per_trial: 4
  gpu_promotion_max_concurrent: 1
  gpu_promotion_gpus_per_trial: 1
  cpu_eval_max_concurrent: 4
```

These defaults are intentionally fair-share rather than cluster-maximal:

- `cpu_screen_max_concurrent: 6` with `4` CPUs each gives this repo up to about `24` CPUs of active screening work
- `gpu_promotion_max_concurrent: 1` reflects the shared single-GPU reality
- `cpu_eval_max_concurrent: 4` leaves room for lightweight post-processing and evaluation jobs without flooding the cluster

The execution block should stop implying one remote trial is enough for hardware utilization. The repo should support:

- multiple CPU trials per iteration
- bounded GPU promotions per iteration
- stage-aware budgets rather than one undifferentiated trial count

## Data flow

1. proposal generation produces candidate configs
2. candidates are assigned a dispatch class: `screen_cpu`, `train_gpu`, or `eval_cpu`
3. CPU screening manifests are emitted first and run broadly
4. screen results are ranked
5. only promoted configs generate GPU manifests
6. GPU results update campaign state and become eligible for final comparison

## Promotion policy

Promotion should be mechanical and explicit, not ad hoc.

Examples:

- top-K CPU configs per iteration
- minimum CPU score threshold
- optional diversity rule so GPU slots are not consumed by near-duplicates

Promotion rules should be stored in config or manifest metadata so results are auditable.

## Implementation boundaries

### `scripts/ray_runner.py`

Keep it as the per-trial execution entrypoint. It already knows how to:

- load config
- train a model
- retry GPU failures
- fall back to CPU

It should not own cluster-wide scheduling policy.

### `scripts/dispatch_trial.py`

This should become the place that expresses job class and generic resource requests:

- CPU screen jobs -> generic CPU remote function
- GPU promotion jobs -> generic GPU remote function
- no named resource labels

### metaopt / manifest generation

Manifest generation should emit job class and resource class explicitly. The current recorded manifest hardcodes:

```json
"resources": {
  "num_gpus": 1,
  "aorus": 1
}
```

That must be replaced with generic GPU requests only.

## Failure handling

- GPU job fails with transient HIP / runtime issue -> retry on GPU
- repeated GPU failure -> demote or fail cleanly according to policy
- CPU saturation -> queue CPU screen jobs; do not bypass directly to GPU unless promotion policy allows it
- queue timeout -> surface clearly in result metadata

## Testing

Add focused tests for:

1. job-class selection
2. manifest resource generation
3. promotion logic
4. generic GPU manifest generation without named resource labels

Smoke checks:

1. one CPU screen dry run
2. one GPU promotion dry run
3. one manifest snapshot test proving absence of host-specific resource names

## Success criteria

The redesign is successful when:

1. this repo no longer emits named-host GPU resource requirements
2. multiple CPU screening jobs can run concurrently on generic Ray capacity
3. GPU work is limited to promoted configs
4. concurrency is bounded and configurable so the project can share the cluster fairly
5. the resulting manifests and code paths make the scheduling intent auditable
