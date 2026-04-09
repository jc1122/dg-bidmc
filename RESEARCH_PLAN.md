# Research Plan: DG-GNN for BIDMC Respiratory Boundary Detection

## Problem Statement

Detect inhale/exhale boundary troughs in ICU impedance pneumography signals,
robust to DC drift, sighs (bursts), and irregular breathing rate — conditions
where statistical SOTA (WaveletDenoise, BinSeg, PELT) degrades significantly.

**Baseline** (zero-shot DG-v5): F1 = 0.322 at ±600 ms tolerance  
**Target**: F1 ≥ 0.70 with trained DG-GNN  
**Ceiling estimate**: F1 ≈ 0.85 (based on annotation quality; annotations are
  inhale peaks, not troughs — some ambiguity is irreducible)

---

## Phase 1: Data Foundation

**Goal**: Build a clean, well-characterized dataset from BIDMC with typed splits.

### 1.1 — Dataset inventory and patient profiling

Characterize all 53 BIDMC patients along four adversarial axes:
- `drift_severity`: first-third vs last-third mean difference / RMS
- `irregularity_cv`: CV of inter-breath intervals
- `sigh_rate`: fraction of breaths with amplitude > 2× median
- `noise_snr`: wavelet-domain SNR estimate

Outputs: `results/patient_profiles.json`

### 1.2 — Ground truth construction

For each patient, extract ground truth breath boundaries:
1. Load impedance pneumography signal, LP-filter at 2 Hz (4th-order Butterworth)
2. Load breath annotations; take every 2nd annotation as inhale peak
3. Trough boundary = `argmin(signal[peak_i : peak_{i+1}])`
4. Reject breaths where `duration < 1.5s` or `duration > 8s` (physiologically impossible)
5. Reject windows where annotation density suggests artifact

Outputs: `results/ground_truth.json` — per-patient list of `{trough_sample, breath_duration, breath_type}`

### 1.3 — Patient split

Split 53 patients into train/val/test, stratified by adversarial profile:
- **Test** (10 patients, held out completely): 2 clean, 2 drift, 2 sighs, 2 irregular, 2 mixed
- **Val** (8 patients): similar stratification  
- **Train** (35 patients)

Rule: no patient appears in more than one split.

Outputs: `results/splits.json`

---

## Phase 2: Feature Engineering

**Goal**: Extract DG graph features per breath window; validate feature quality.

### 2.1 — Graph extraction pipeline

For each 6-breath window (with ±30-sample padding):
1. LP-filter the signal (2 Hz)
2. Detrend (linear, window-local)
3. Call `build_graph(signal)` → `motifs.EdgeGraph`
4. Extract node features:
   - `bar_position` (normalized 0–1 within window)
   - `amplitude` (abs value at node)
   - `duration` (samples to next node)
   - `direction` (up=1 / down=−1, from sign of amplitude change)
   - `log_edge_size` (log of `edge_size`, cast to float64 first)
   - `level` (from `assign_levels()` KMeans, n_levels=3)
5. Edge features:
   - `log_edge_size` of the connecting edge
   - `direction_match` (1 if src and dst have same direction)

### 2.2 — Node label assignment

Label each DG node as:
- `boundary=1` if within ±`TOL` samples of a ground-truth trough
- `boundary=0` otherwise
- TOL = 75 samples (±600 ms)

Additionally, label boundary nodes with breath type:
- `normal` (within ±20% of median duration and amplitude)
- `sigh` (amplitude > 2× median)
- `irregular` (duration outside ±40% of median)

### 2.3 — Feature validation

Visualize feature distributions:
- Plot `log_edge_size` histograms for boundary vs non-boundary nodes
- Compute mutual information between each feature and `boundary` label
- Compute Spearman correlation between `log_edge_size` level and GT boundary status

Expected: level-0 (coarsest) DG nodes are much more likely to be boundaries.

---

## Phase 3: Model

**Goal**: Train a GAT that classifies DG nodes as boundary / non-boundary.

### 3.1 — Architecture: DG-GAT

```python
class DGGAT(torch.nn.Module):
    def __init__(self, in_dim=6, hidden=64, heads=8, n_layers=3):
        # GATConv layers with residual connections
        # Multi-head output
        pass

    def forward(self, x, edge_index, edge_attr):
        # Node classification head: boundary score per node
        # Breath type head: classification (normal/sigh/irregular) on boundary nodes
        # Rate regression head: global mean-pooled → breaths/min estimate
        pass
```

Library: PyTorch Geometric (`torch_geometric`) — already installed system-wide.

Loss:
- Boundary classification: weighted BCE (positive weight ~5× to handle class imbalance)
- Type classification: cross-entropy (only on predicted boundary nodes)
- Rate regression: smooth L1

### 3.2 — Training setup

- **Hardware**: Aorus GPU via Ray remote task
- **Data loading**: pre-extract all graphs from train split; cache as `.pt` files
- **Batch size**: 32 windows (variable graph sizes → use `torch_geometric.data.DataLoader`)
- **Optimizer**: AdamW, lr=1e-3, weight decay 1e-4
- **Schedule**: CosineAnnealingLR, T_max=100 epochs
- **Early stopping**: val boundary F1, patience=15

### 3.3 — Ablations

Run on val set after convergence:
1. DG features only (no preprocessing) vs DG + LP filter
2. 1-layer vs 2-layer vs 3-layer GAT
3. Edge features vs no edge features
4. KMeans levels as feature vs not (does hierarchical level assignment add signal?)
5. Freeze DG graph topology at train time vs recompute per epoch (deterministic, so same)

---

## Phase 4: Evaluation

**Goal**: Honest comparison against published and classical methods.

### 4.1 — Metrics

Primary:
- **Boundary F1** at ±600 ms tolerance (matches zero-shot baseline)
- **Boundary F1** at ±300 ms tolerance (stricter, more clinically relevant)

Secondary:
- **Respiratory rate MAE** (breaths/min) — compute from predicted boundaries
- **Breath type accuracy** (normal/sigh/irregular classification)
- **Robustness delta**: F1_irregular_rate − F1_regular (DG should be stable; WD degrades)

### 4.2 — Comparison methods

| Method | Type | Notes |
|--------|------|-------|
| DG-GAT (ours) | Trained structural | Target ≥ 0.70 F1 |
| DG-v5 zero-shot | Zero-shot structural | 0.322 F1 (established baseline) |
| WaveletDenoise | Classical untrained | 0.291 F1 (best zero-shot SOTA) |
| BinSeg | Classical untrained | 0.192 F1 |
| CNN-1D trained | Trained baseline | Train same data as DG-GAT for fair comparison |
| PPG2RespNet style | Trained deep (UNet) | Reference from literature (different signal/task) |

### 4.3 — Adversarial breakdown

Report per-condition F1 (test set only):
- Regular breathing
- DC drift (mild: <5% RMS, strong: >10% RMS)
- Sighs (windows containing ≥1 sigh)
- Irregular rate (CV > 0.20)
- Mixed (≥2 adversarial conditions present)

---

## Phase 5: Stretch Goals

If Phase 4 target is met (F1 ≥ 0.70):

### 5.1 — Apnea detection

Central apnea = absence of breath boundaries for > 10 s. DG-GAT output:
if no high-confidence boundary node in a window → flag as apnea candidate.
Evaluate against any available apnea labels in PhysioNet.

### 5.2 — Cross-signal transfer

Test DG-GAT (trained on impedance pneumography) on PPG-derived respiratory signal
from BIDMC (`rec.p_signal[:, 1]`). Expect degradation; measure how much.
Fine-tune with 5 patients → measure recovery.

### 5.3 — Online / streaming inference

DG graph construction is O(n). Profile end-to-end latency on Aorus for
real-time (125 Hz) inference: can DG-GAT run at ≥10 Hz update rate?

---

## Success Criteria

| Criterion | Target | Stretch |
|-----------|--------|---------|
| Boundary F1 @600ms (test set) | ≥ 0.70 | ≥ 0.80 |
| Boundary F1 @300ms (test set) | ≥ 0.55 | ≥ 0.70 |
| Irregular-rate robustness delta over WD | ≥ +0.10 | ≥ +0.20 |
| Rate MAE (breaths/min) | ≤ 2.0 | ≤ 1.0 |
| Training patients needed | ≤ 35 | ≤ 15 |

---

## Implementation Order

```
Phase 1.1 → 1.2 → 1.3      (data, ~1 day)
Phase 2.1 → 2.2 → 2.3      (features, ~1 day)
Phase 3.1 → 3.2             (model + training, ~2 days on Aorus)
Phase 3.3                   (ablations, ~1 day)
Phase 4.1 → 4.2 → 4.3      (evaluation, ~1 day)
Phase 5.x                   (stretch, as time allows)
```

Key dependency: Phase 3.2 requires Aorus GPU (Hetzner head + Aorus attached).
Phases 1–2 and all evaluation scripts can run on laptop.

---

## Files to Reuse from `dg_constituency_benchmark`

| File | What to reuse |
|------|--------------|
| `scripts/dg_pipeline.py` | `build_graph()`, `assign_levels()` — copy or import |
| `scripts/dg_pipeline_v2.py` | `_suppress_bursts()`, `_denoise_wavelet()`, `_remove_drift()` |
| `scripts/common.py` | `boundary_f1()`, `boundary_precision()`, `boundary_recall()`, `timer()` |
| `scripts/adversarial.py` | `add_dc_drift()`, `add_burst_noise()` — for training augmentation |
| `results/real_data_respiratory.json` | Zero-shot baseline numbers for comparison table |
