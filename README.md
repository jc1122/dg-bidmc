# DG-BIDMC: Respiratory Boundary Detection with DegreeGraph + GNN

Applies the DegreeGraph structural parsing approach — validated on synthetic
SCFG signals in `dg_constituency_benchmark` — to real ICU respiratory signals
from the BIDMC PhysioNet dataset.

**Core thesis**: DegreeGraph's zero-crossing graph is the natural structural
skeleton of a respiratory signal. A lightweight GNN trained on that graph can
detect inhale/exhale boundaries with fewer labels and better adversarial
robustness than statistical SOTA methods.

---

## Background: What We Learned from the Constituency Benchmark

The `dg_constituency_benchmark` project validated DegreeGraph as a constituency
parser for SCFG-generated waveforms. Key findings that directly apply here:

### DegreeGraph structural advantages (confirmed empirically)

| Condition | DG-v5 vs best SOTA | Why DG wins |
|-----------|-------------------|-------------|
| DC drift | +0.416 (synthetic) / +0.009 (real) | Drift-invariant: DG uses relative turning points, not absolute amplitude |
| Burst / impulsive noise | +0.157 (synthetic) / +0.001 (real) | Burst suppression stage clips outliers before graph construction |
| Irregular rate variation | +0.124 (real BIDMC) | Structural parsing adapts to locally variable segment lengths |
| Compound (drift+noise+harmonics) | +0.107 (synthetic) | Self-gating pipeline isolates degradation types |

### Where DG-v5 loses to WaveletDenoise (to keep in mind)

| Condition | WD advantage | Root cause |
|-----------|-------------|------------|
| Stationary additive noise (white/pink) | −0.124 ratio / −0.059 varamp | KMeans saturation on uniform amplitude |
| Regular high-SNR signals | small | WD's spectral prior matches |

### The DG pipeline (directly reusable)

```python
# From dg_pipeline.py — zero dependencies beyond degreegraph + motifs
graph = build_graph(signal)          # → motifs.EdgeGraph
node_level, edge_level = assign_levels(graph, n_levels=3)  # KMeans on log(edge_size)
```

The pipeline in `dg_pipeline_v2.py` adds self-gating preprocessing:
1. Burst suppression (MAD-based outlier clipping)
2. Harmonic removal (FFT notch filtering)
3. Wavelet denoising (two-tier VisuShrink)
4. Drift removal (conditional linear detrend)
5. Standard DG parse

For respiratory signals at 125 Hz: add **LP filter at 2 Hz** before DG
(removes sub-breath oscillations; confirmed effective in BIDMC tests).

### Key parameter: D_MIN

`D_MIN` is the minimum segment duration in samples. For respiratory at 125 Hz:
- Typical breath period: 4–6 s = 500–750 samples
- Half-breath (inhale or exhale): 250–375 samples
- D_MIN = 55% × half-breath ≈ 130–200 samples
- Use 120–160 in practice; too small → over-segmentation, too large → merges breaths

### Real-data baseline (zero-shot DG-v5 on BIDMC, n=5 patients, 8 windows each)

| Condition | BinSeg | WaveletDenoise | DG-v5 | Δ DG vs SOTA |
|-----------|--------|---------------|-------|-------------|
| Regular | 0.188 | 0.272 | 0.283 | +0.011 |
| DC drift | 0.292 | 0.431 | 0.429 | ≈ 0 |
| Strong drift | 0.188 | 0.343 | 0.363 | +0.020 |
| Sighs (burst) | 0.250 | 0.274 | 0.275 | +0.001 |
| Irregular rate | 0.042 | 0.138 | 0.261 | **+0.124** |
| **Mean** | **0.192** | **0.291** | **0.322** | **+0.031** |

F1 tolerance: ±600 ms. These are **zero-shot** results — DG was never trained on
respiratory data. With domain-specific GNN training, we expect 0.65–0.85.

### Current rigorous evaluation protocol

The project now uses a stricter reporting split between **model selection** and
**final comparison**:

| Stage | Metric | Purpose |
|------|--------|---------|
| Validation | Boundary F1 @ ±600 ms | Hyperparameter / architecture selection |
| Held-out test | Boundary F1 @ ±600 ms | Main comparison metric |
| Held-out test | Patient-macro F1 @ ±600 ms | Prevent patient imbalance from hiding failures |
| Held-out test | Boundary F1 @ ±300 ms | Stricter timing-sensitive metric |
| Held-out test | Event precision / recall | Error mode analysis |
| Held-out test | Boundary timing MAE (ms) | Localization quality |
| Held-out test | Rate MAE / RMSE / bias / LoA / correlation | RR literature comparison |

**Important:** there is no published external per-breath trough-boundary benchmark
for BIDMC impedance pneumography with this exact matching protocol. For boundary
detection, the rigorous same-task baselines are therefore the internal classical
methods above (**DG-v5 0.322, WaveletDenoise 0.291, BinSeg 0.192**). Published
literature comparisons are most rigorous for the **rate** metrics, not for the
exact trough-boundary F1 task.

The preserved best validation checkpoint currently reproduces:
- **Validation:** F1@600ms = 0.9792, rate MAE = 0.198 bpm
- **Held-out test:** F1@600ms = 0.9110, rate MAE = 0.5149 bpm

So the honest generalization estimate is the **held-out test** score, not the
validation peak.

---

## Dataset

**BIDMC PhysioNet Respiratory Dataset**  
53 ICU patients, each 8-minute recording.
- Signal: impedance pneumography (thoracic electrical impedance), 125 Hz
- Annotations: breath peaks (`wfdb.rdann(..., 'breath')`, symbol `'"'`)
  - Every 2nd annotation = inhale peak; ground-truth boundaries = troughs between peaks
- Also available: PPG, SpO2, ECG (not used in this project)

```python
import wfdb
rec = wfdb.rdrecord('bidmc01', pn_dir='bidmc')
ann = wfdb.rdann('bidmc01', 'breath', pn_dir='bidmc')
resp = rec.p_signal[:, 0]   # impedance pneumography channel
peaks = ann.sample[::2]      # inhale peaks (every 2nd annotation)
```

Patients vary in:
- **Breathing regularity** (CV of inter-breath interval)
- **Baseline drift** (linear drift amplitude)
- **Sighs** (occasional large-amplitude breaths)
- **Rate** (~6–25 breaths/min range across patients)

---

## Technical Approach: DG-GNN

### Why a GNN on the DG graph?

The DG graph from a respiratory signal has:
- **Nodes** = zero-crossing segments (inhale/exhale arches)
- **Edges** = structural connections (forward edges only, src_bar < dst_bar)
- **Node features**: position, amplitude, duration, direction (up/down)
- **Edge features**: `edge_size` (hierarchical level indicator)

This is a natural graph for a GNN:
- Fewer nodes than raw signal samples (10–50 nodes per 6-breath window vs 500–2000 samples)
- Structural prior built in — the GNN starts from a skeleton, not noise
- Graph attention (GAT) can learn to suppress false positive boundaries

### Architecture: DG-GAT

```
Input: raw respiratory signal (N samples)
   ↓
Preprocessing (LP filter, detrend)
   ↓
DG graph construction: build_graph() → EdgeGraph
   ↓
Node features: [bar_position, amplitude, duration, direction, log_edge_size]
   ↓
GAT layers (2–3 layers, 8 attention heads)
   ↓
Multi-head output:
  ├── Boundary score (0/1 per node) → F1 metric
  ├── Breath type (normal / sigh / irregular) → classification accuracy
  └── Rate regression (breaths/min) → MAE
```

### Why this beats pure deep learning

| Property | PPG2RespNet (SOTA UNet) | DG-GAT |
|----------|------------------------|--------|
| Input representation | Raw signal (dense) | Graph (structural skeleton) |
| Training data needed | All 53 patients | ~10–20 patients |
| Adversarial robustness | Unknown (trained on clean ICU) | Structural, inherits DG robustness |
| Interpretability | Black box | Each node corresponds to a physical segment |
| Cross-sensor transfer | PPG-specific | Signal-agnostic (any oscillatory signal) |

---

## External Libraries (pre-installed)

- `degreegraph` — `compute_arrays(highs, lows)` → CSR arrays
  - Source: `/home/jakub/projects/DegreeGraph2` (editable install)
- `motifs` — `EdgeGraph.from_degreegraph2(...)` → graph with `edge_size`, `node_bar`, etc.
  - `edge_size` is `float32` — cast to `float64` before log-transform

Both are installed system-wide. Do not add to requirements.txt.

---

## Infrastructure

```
Laptop (control plane, code editing, light tests)
    │
    ▼
ray-head @ 89.167.79.237 (Hetzner cx53, 16 CPU, Ray 2.40.0)
    │
    └── Aorus @ 100.104.191.44 (AMD RX 5700 XT, ROCm, 16 CPU)
         torch 2.9.1a0+gitd38164a — NEVER pip-install/upgrade torch here
```

### Dispatch model: generic resources only

This project uses **staged dispatch** with generic Ray resource requests:

| Stage | Resource request | Purpose |
|-------|-----------------|---------|
| `screen_cpu` | `num_cpus: N` | Cheap/fast CPU screening trials |
| `train_gpu` | `num_gpus: N` | Full GPU training for promoted trials |
| `eval_cpu` | `num_cpus: 1` | Final evaluation on CPU |

The project **never** encodes host identity (e.g. named custom resources) in
dispatch manifests. Host-to-resource mapping is handled externally by the
Ray cluster scheduler and the `ray-hetzner` autoscaler configuration.

```python
# Project-side dispatch (scripts/dispatch_trial.py):
ray.remote(num_cpus=2)(screen_fn)      # screen_cpu stage
ray.remote(num_gpus=1)(train_fn)       # train_gpu stage — scheduler places on GPU node
```

Remote batch execution is delegated through the Hetzner queue workflow.
Always pass the explicit per-project queue root
`/home/jakub/projects/dg_bidmc/.ml-metaopt`. If that root is not already in the
daemon's watched set, enqueue must be followed by a one-shot reconcile on the
Aorus head; otherwise the batch will remain queued but idle.
See `~/projects/ray-hetzner/CLAUDE.md` for cluster setup details.

---

## Related Projects

| Project | Relationship |
|---------|-------------|
| `dg_constituency_benchmark` | Origin: validated DG on synthetic SCFG signals; produced DG-v5 pipeline |
| `dg_signal_benchmark` | Validated DG+Motifs for component separation and burst detection (GNN direction confirmed there) |
| `DegreeGraph2` | Core graph construction library |
| `Motifs` | EdgeGraph API, forward-edge metadata, `edge_size` |

See `dg_constituency_benchmark/results/real_data_respiratory.json` for zero-shot baseline.

---

## See Also

- `RESEARCH_PLAN.md` — phased implementation plan
- `scripts/` — data loading, feature extraction, model, evaluation
- `results/` — experiment outputs
