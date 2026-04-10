# DG-BIDMC Agent Context

This file provides complete context for a coding agent working on the
`dg_bidmc` project. Read this entire file before writing any code.

---

## Project Goal

Build a **DG-GAT** model that detects inhale/exhale trough boundaries in ICU
respiratory signals from the BIDMC PhysioNet dataset. The approach:

1. Preprocess the impedance pneumography signal (LP filter, detrend)
2. Build a DegreeGraph structural graph (`build_graph()`)
3. Extract node/edge features from the graph
4. Train a Graph Attention Network (GAT) to classify nodes as boundaries

**Baseline** (DG-v5 zero-shot, no training): F1 = 0.322 @ ±600ms tolerance  
**Selection metric**: validation boundary F1 @ ±600ms tolerance  
**Final reporting**: held-out test boundary F1 @ ±600ms, patient-macro F1 @ ±600ms,
strict F1 @ ±300ms, and RR MAE/RMSE/bias/LoA

---

## What DegreeGraph Is

DegreeGraph2 is a structural signal parsing library that builds a hierarchical
graph from a time series via zero-crossing analysis.

```python
import degreegraph
import numpy as np
from motifs import EdgeGraph

signal = np.asarray(signal, dtype=np.float64)
indices, is_lows, offsets, connections = degreegraph.compute_arrays(signal, signal)
graph = EdgeGraph.from_degreegraph2(indices, is_lows, offsets, connections, signal)
```

Key graph attributes:
- `graph.node_bar` — sample index of each node (turning point)
- `graph.node_value` — signal amplitude at each node
- `graph.edge_size` — hierarchical importance of each edge (float32 — ALWAYS cast to float64 before log)
- `graph.edge_source_rows` — source node index for each edge
- `graph.edge_to` — destination node index for each edge
- Forward edges only: `edge_source_rows[i] < edge_to[i]` in bar space

**CRITICAL**: `edge_size` is float32. Cast before log: `np.asarray(graph.edge_size, dtype=np.float64)`

### What graph nodes represent

Each node is a **turning point** (local extremum) in the signal. For a
respiratory signal:
- Nodes at inhale peaks (local maxima) and exhale troughs (local minima)
- The **troughs are our ground-truth boundary locations**
- Large `edge_size` → coarse structural transition (likely breath boundary)
- Small `edge_size` → fine oscillation (noise or sub-breath structure)

### KMeans level assignment

```python
from sklearn.cluster import KMeans
log_es = np.log(np.asarray(graph.edge_size, dtype=np.float64) + 1e-8).reshape(-1, 1)
km = KMeans(n_clusters=3, n_init=10, random_state=42)
edge_level = km.fit_predict(log_es)
# Level 0 = coarsest (largest edge_size) = most likely breath boundary
```

This is implemented in `scripts/dg_pipeline.py::assign_levels()`.

---

## Dataset: BIDMC PhysioNet

53 ICU patients, 8-minute recordings at 125 Hz.

```python
import wfdb
rec = wfdb.rdrecord('bidmc01', pn_dir='bidmc')   # downloads on first use
ann = wfdb.rdann('bidmc01', 'breath', pn_dir='bidmc')
resp = rec.p_signal[:, 0]    # impedance pneumography (channel 0)
fs = rec.fs                   # 125 Hz
peaks = ann.sample[::2]       # inhale peaks (every 2nd annotation)
```

**Ground truth construction**:
```python
from scipy.signal import butter, filtfilt
b, a = butter(4, 2.0 / (fs/2), 'low')
resp_lp = filtfilt(b, a, resp - resp.mean())
# Trough = argmin between consecutive inhale peaks
troughs = []
for i in range(len(peaks)-1):
    p1, p2 = int(peaks[i]), int(peaks[i+1])
    trough = p1 + int(np.argmin(resp_lp[p1:p2]))
    troughs.append(trough)
```

**Key preprocessing for respiratory signals**:
- Always LP-filter at 2 Hz (removes sub-breath noise, reduces zero-crossing density)
- Always detrend (linear, window-local) before DG graph construction
- D_MIN ≈ 55% × mean_half_breath_period, clamped to [50, 160] samples

**Patient profiles** (measured from 9 patients):

| Patient | Condition | Drift CV | Sighs | Notes |
|---------|-----------|----------|-------|-------|
| bidmc01 | drift | 0.127 | 2 | strongest drift (0.034 RMS ratio) |
| bidmc02 | burst | 0.153 | 4 | most sighs |
| bidmc03 | drift | 0.144 | 3 | 2nd most drift (0.038) |
| bidmc04 | noise | 0.193 | 3 | most irregular rate (highest CV) |
| bidmc05 | slow | — | — | very slow: ~6 bpm, only 48 breaths |
| bidmc06 | clean | 0.109 | 2 | most regular (lowest CV) |
| bidmc07 | clean | 0.117 | 2 | regular |
| bidmc08 | moderate | 0.151 | 3 | moderate |

---

## Zero-Shot Baseline Results (DG-v5, No Training)

From `results/real_data_respiratory.json`. 5 patients × 8 windows × 6 breaths:

| Patient | Condition | BinSeg | WaveletDenoise | DG-v5 |
|---------|-----------|--------|---------------|-------|
| bidmc06 | clean | 0.188 | 0.272 | 0.283 |
| bidmc01 | dc_drift | 0.292 | 0.431 | 0.429 |
| bidmc03 | strong drift | 0.188 | 0.343 | 0.363 |
| bidmc02 | burst/sighs | 0.250 | 0.274 | 0.275 |
| bidmc04 | irregular | 0.042 | 0.138 | **0.261** |
| **Mean** | | **0.192** | **0.291** | **0.322** |

DG-v5 beats SOTA on all condition types. Biggest win: irregular rate (+0.124 over WD).

---

## Synthetic Benchmark Results (DG-v5-ensemble, n=30)

`results/dg_v5_synthetic_benchmark.json` — composite F1 = 0.7807 across 39
adversarial conditions on synthetic SCFG signals at 2000 Hz.

Key condition scores relevant to respiratory:
- burst_snr20: 0.885, burst_snr0: 0.904 → very robust to impulsive noise
- dc_drift_0.005: 0.920 → excellent drift handling
- compound_mild: 0.566, compound_severe: 0.492 → compound conditions are hardest
- harmonics_clean: 0.699 → moderate harmonic interference tolerance

These show where DG excels (burst, drift) and struggles (compound, harmonics).

---

## Code Architecture (What's in `scripts/`)

### `scripts/dg_pipeline.py` — Core DG primitives (from constituency benchmark)
- `build_graph(signal)` → `motifs.EdgeGraph`
- `assign_levels(graph, n_levels=3)` → `(node_level, edge_level)` arrays
- `recover_parse_tree(graph, node_level, signal)` → structured parse result

### `scripts/dg_pipeline_v2.py` — Self-gating preprocessing pipeline
Functions reusable for BIDMC:
- `_suppress_bursts(signal, mad_multiplier=6.0)` — clip impulsive outliers
- `_suppress_harmonics(signal, fs=2000)` — FFT notch filtering
- `_denoise_wavelet(signal, wavelet='db4')` — VisuShrink denoising
- `_remove_drift(signal)` — conditional linear detrend
- `preprocess_signal(signal, fs=2000)` — runs all 4 stages

**NOTE**: `dg_pipeline_v2.py` uses `FS=2000` internally. For respiratory (125 Hz),
call the individual stage functions directly, not `preprocess_signal()`.

### `scripts/common.py` — Shared metrics and helpers (from constituency benchmark)
- `boundary_f1(pred_starts, pred_ends, true_starts, true_ends, ...)` — primary metric
- `boundary_precision(...)`, `boundary_recall(...)` — components
- `timer()` — context manager for timing
- **NOTE**: `FS=2000`, `D_MIN=200` — these constants are for the synthetic benchmark;
  do not import them for BIDMC use

### `scripts/adversarial.py` — Noise injection for training augmentation
- `add_dc_drift(signal, drift_rate)` — add linear drift
- `add_burst_noise(signal, snr_db, rng)` — add impulsive noise
- `add_pink_noise(signal, snr_db, rng)` — add pink noise
- `add_brown_noise(signal, snr_db, rng)` — add brown noise
- `add_harmonic_interference(signal, ...)` — add tonal interference

Use these for **data augmentation** during DG-GAT training.

### `scripts/baseline_wavelet_denoise.py` — WaveletDenoise baseline
For comparison. Uses `D_MIN//3` as merge threshold.

---

## DG-GAT Architecture

Build this in `scripts/model.py`:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class DGGAT(torch.nn.Module):
    """Graph Attention Network on DegreeGraph structural graph."""

    def __init__(self, in_dim: int = 6, hidden: int = 64, heads: int = 8,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden, heads=heads, dropout=dropout,
                                  edge_dim=2, concat=True))
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads,
                                      dropout=dropout, edge_dim=2, concat=True))
        self.convs.append(GATConv(hidden * heads, hidden, heads=1,
                                  dropout=dropout, edge_dim=2, concat=False))

        # Multi-head output
        self.boundary_head = torch.nn.Linear(hidden, 1)     # per-node boundary score
        self.type_head = torch.nn.Linear(hidden, 3)         # normal/sigh/irregular
        self.rate_head = torch.nn.Linear(hidden, 1)         # global rate regression

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index, edge_attr))
        x = self.convs[-1](x, edge_index, edge_attr)

        boundary_logits = self.boundary_head(x).squeeze(-1)  # [N]
        type_logits = self.type_head(x)                       # [N, 3]
        rate_pred = self.rate_head(x).mean(dim=0)             # scalar

        return boundary_logits, type_logits, rate_pred
```

**Node features** (in_dim=6):
1. `bar_position_norm` — sample index / window_length (0–1)
2. `amplitude_norm` — signal value at node, z-score normalized within window
3. `duration_norm` — samples to next node / mean_breath_length
4. `direction` — +1 for peak (local max), −1 for trough (local min)
5. `log_edge_size_norm` — log(edge_size + 1e-8), z-score normalized
6. `level` — KMeans cluster level (0=coarsest, 2=finest), normalized /2

**Edge features** (edge_dim=2):
1. `log_edge_size_norm` — same normalization as node feature
2. `direction_match` — 1 if src and dst have same direction, 0 otherwise

---

## Graph-to-Data Construction

Build this in `scripts/graph_features.py`:

```python
import numpy as np
import torch
from torch_geometric.data import Data

def extract_graph_data(signal: np.ndarray, gt_troughs: list[int],
                       tol_samples: int = 75) -> Data:
    """Convert DG graph to PyG Data object with node labels."""
    from dg_pipeline import build_graph, assign_levels

    # Build graph
    graph = build_graph(signal)
    node_level, edge_level = assign_levels(graph, n_levels=3)

    # Node positions and values
    node_bars = np.asarray(graph.node_bar)
    node_vals = np.asarray(graph.node_value, dtype=np.float64)
    n_nodes = len(node_bars)

    # Directions: +1 = peak (local max = inhale), -1 = trough (local min = exhale)
    direction = np.where(node_vals >= 0, 1.0, -1.0)

    # Duration: samples between consecutive nodes
    duration = np.diff(node_bars, append=node_bars[-1] + 1).astype(float)

    # Normalize within window
    N = len(signal)
    bar_norm = node_bars / N
    amp_norm = (node_vals - node_vals.mean()) / (node_vals.std() + 1e-8)
    dur_norm = duration / (duration.mean() + 1e-8)

    es = np.asarray(graph.edge_size, dtype=np.float64)
    log_es_edge = np.log(es + 1e-8)
    log_es_norm = (log_es_edge - log_es_edge.mean()) / (log_es_edge.std() + 1e-8)

    # Per-node log_edge_size: max of connected edges
    log_es_node = np.full(n_nodes, log_es_norm.min())
    src = np.asarray(graph.edge_source_rows)
    dst = np.asarray(graph.edge_to)
    for i, (s, d) in enumerate(zip(src, dst)):
        log_es_node[s] = max(log_es_node[s], log_es_norm[i])
        log_es_node[d] = max(log_es_node[d], log_es_norm[i])

    # Node features [N, 6]
    x = np.stack([bar_norm, amp_norm, dur_norm, direction,
                  log_es_node, node_level / 2.0], axis=1).astype(np.float32)

    # Edge index and features
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    dir_match = (direction[src] == direction[dst]).astype(np.float32)
    edge_attr = np.stack([log_es_norm, dir_match], axis=1).astype(np.float32)

    # Node labels: boundary if within tol of any GT trough
    y_boundary = np.zeros(n_nodes, dtype=np.float32)
    for trough in gt_troughs:
        dists = np.abs(node_bars - trough)
        if dists.min() <= tol_samples:
            y_boundary[np.argmin(dists)] = 1.0

    return Data(
        x=torch.tensor(x),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr),
        y=torch.tensor(y_boundary),
        node_bars=torch.tensor(node_bars),
        n_gt_troughs=len(gt_troughs),
    )
```

---

## Training Details

- **Library**: PyTorch Geometric (`torch_geometric`) — install if missing: `pip install torch_geometric`
- **Hardware**: Aorus GPU for training (see Infrastructure section)
- **Data**: cache all graph `.pt` files from train split before training loop
- **Class imbalance**: ~1 boundary node per 5–10 total nodes → use `pos_weight=7.0` in BCEWithLogitsLoss
- **Batch size**: 32 graphs (use `torch_geometric.loader.DataLoader`)
- **Optimizer**: AdamW(lr=1e-3, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR(T_max=100)
- **Early stopping**: patience=15, monitor val boundary F1

**Loss function**:
```python
bce = F.binary_cross_entropy_with_logits(boundary_logits, y_boundary,
                                          pos_weight=torch.tensor(7.0))
# Add type_loss and rate_loss only when boundary GT available
total_loss = bce + 0.3 * type_loss + 0.1 * rate_loss
```

---

## Evaluation

Primary selection metric: **boundary F1 at ±600ms** (75 samples at 125 Hz)

Final reporting metrics:
- **Boundary F1 @ ±600ms** (graph-macro)
- **Patient-macro Boundary F1 @ ±600ms**
- **Boundary F1 @ ±300ms** (38 samples at 125 Hz)
- **Event precision / recall @ ±600ms**
- **Matched-boundary timing MAE**
- **Respiratory-rate MAE / RMSE / bias / 95% LoA / correlation**

Rigour note: no published external per-breath trough-boundary benchmark was found
for BIDMC impedance pneumography with this exact event-matching protocol. Compare
boundary detection against same-protocol internal baselines (DG-v5, WaveletDenoise,
BinSeg) and compare RR metrics contextually against the published RR literature.

```python
# From scripts/common.py — use directly
from common import boundary_f1

# Convert predicted boundary node bars to starts/ends arrays
pred_troughs = node_bars[boundary_scores > threshold]
f1 = boundary_f1(
    pred_starts=pred_troughs,
    pred_ends=pred_troughs + 1,
    true_starts=np.array(gt_troughs),
    true_ends=np.array(gt_troughs) + 1,
    true_durations=np.full(len(gt_troughs), mean_breath_len),
    tolerance_frac=0.0,
    tolerance_floor=75,   # ±600ms at 125Hz
)
```

Compare against:
1. DG-v5 zero-shot (numbers in `results/real_data_respiratory.json`)
2. WaveletDenoise (import from `scripts/baseline_wavelet_denoise.py`)
3. BinSeg (`ruptures.Binseg`)
4. CNN-1D trained baseline (implement as fair comparison)

---

## Infrastructure

```
Laptop (code editing, light tests, data loading)
    │
    ▼
ray-head @ 89.167.79.237 / 100.105.11.68 (Tailscale)
    │  Hetzner cx53, 16 CPU, Ray 2.40.0
    │  Project synced to: /root/dg_bidmc/
    │  ray-hetzner repo: /root/ray-hetzner/
    │
    └── Aorus @ 100.104.191.44 (Tailscale)
         AMD RX 5700 XT, ROCm, 16 CPU
         torch 2.9.1a0+gitd38164a — NEVER pip-install/upgrade torch here
         System Python: /usr/bin/python3
         Ray venv: ~/ray-venv (Ray only, no torch)
```

**NEVER** run `pip install torch` or `pip install --upgrade torch` on Aorus.
The ROCm torch build is custom; reinstalling will break GPU access.

### Dispatch contract: generic resources only

The project dispatch path (`scripts/dispatch_trial.py`,
`scripts/build_batch_manifest.py`) uses **only generic Ray resources**:

| Stage | Resource request | Purpose |
|-------|-----------------|---------|
| `screen_cpu` | `{num_cpus: N}` | Cheap CPU screening trials |
| `train_gpu` | `{num_gpus: N}` | Full GPU training for promoted configs |
| `eval_cpu` | `{num_cpus: 1}` | Final CPU evaluation |

**No named custom resources** (e.g. `resources={"aorus": 1}`) appear in the
project dispatch path. Host placement is delegated to the Ray scheduler and
the `ray-hetzner` autoscaler configuration. This keeps the project portable:
adding or replacing GPU nodes requires only cluster-side config changes.

To run GNN training via the project dispatcher:
```python
# Generic resource dispatch — scheduler places on any GPU node
from scripts.dispatch_trial import dispatch
dispatch(config, project_root, output_path)
```

To push code to head node:
```bash
cd ~/projects/ray-hetzner
./push_code.sh ~/projects/dg_bidmc /root/dg_bidmc
```

---

## File Layout

```
dg_bidmc/
├── AGENTS.md               ← this file
├── CLAUDE.md               ← infrastructure + session resume
├── README.md               ← project overview and background
├── RESEARCH_PLAN.md        ← 5-phase implementation plan
├── requirements.txt        ← pip dependencies
├── scripts/
│   ├── dg_pipeline.py      ← build_graph(), assign_levels() [copied from constituency benchmark]
│   ├── dg_pipeline_v2.py   ← self-gating preprocessing [copied]
│   ├── common.py           ← boundary_f1(), timer() [copied — note FS=2000 constants, ignore]
│   ├── adversarial.py      ← noise injection for augmentation [copied]
│   ├── baseline_wavelet_denoise.py  ← WD baseline [copied]
│   ├── baseline_pelt.py    ← PELT baseline [copied — hangs on noisy signals, use with timeout]
│   ├── data_loader.py      ← [TO CREATE] BIDMC loading, GT construction, splits
│   ├── graph_features.py   ← [TO CREATE] DG graph → PyG Data objects
│   ├── model.py            ← [TO CREATE] DGGAT architecture
│   ├── train.py            ← [TO CREATE] training loop
│   └── evaluate.py         ← [TO CREATE] evaluation harness
├── results/
│   ├── real_data_respiratory.json   ← zero-shot DG-v5 baseline results
│   └── dg_v5_synthetic_benchmark.json ← synthetic benchmark results (for reference)
└── tests/
    └── test_graph_features.py       ← [TO CREATE]
```

---

## Common Pitfalls

1. **`edge_size` is float32** — always cast to float64 before log-transform
2. **PELT hangs** on noisy signals with >200 zero crossings — use BinSeg instead
3. **D_MIN must be adapted** to the signal sampling rate — for respiratory at 125 Hz,
   use 120–160 samples (not the default 200 from the constituency benchmark)
4. **Don't import FS or D_MIN from common.py** — those are for the 2000 Hz SCFG benchmark
5. **`preprocess_signal()` in dg_pipeline_v2.py uses FS=2000** — call stage functions
   individually for respiratory signals
6. **LP filter before DG** — always apply `butter(4, 2.0/(125/2), 'low')` to respiratory
   signals before calling `build_graph()`; this reduces zero-crossing density dramatically
7. **torch_geometric not installed on Aorus** by default — may need `pip install torch_geometric`
   in the Ray venv (not the system Python)
8. **GATConv must use `add_self_loops=False` on Aorus GPU** — the RX 5700 XT (gfx1010) custom
   ROCm torch build lacks a compiled kernel for `torch.nonzero()` on GPU; GATConv's default
   `add_self_loops=True` internally calls `remove_self_loops` which uses boolean tensor indexing
   → `hipErrorInvalidDeviceFunction`. Since DG graphs are DAGs with no self-loops, setting
   `add_self_loops=False` is both correct and avoids the broken code path.
   **Patch opportunity**: `jc1122/pytorch-gfx1010` covers composable_kernel (matmul/conv) and
   BatchNorm2d backward (MIOpen DPP). The `torch.nonzero()` / boolean-index kernel gap is NOT
   yet covered. When patching, add a HIP kernel for `at::nonzero_cuda` compiled for `gfx1010`.
   Until then, avoid all GPU boolean-index ops (nonzero, masked_select, bool-indexed __getitem__).
9. **`build_graph()` requires C-contiguous float64 input** — `scipy.signal.filtfilt()` output is
   not C-contiguous. Always use `np.ascontiguousarray(signal, dtype=np.float64)` before passing
   to `degreegraph.compute_arrays()` or `build_graph()`.

---

## Implementation Order

Follow `RESEARCH_PLAN.md`. The implementation sequence is:

1. `scripts/data_loader.py` — load BIDMC, extract GT troughs, create splits
2. `scripts/graph_features.py` — DG → PyG Data, feature extraction, node labeling
3. `scripts/model.py` — DGGAT architecture
4. `scripts/train.py` — training loop, checkpointing
5. `scripts/evaluate.py` — full evaluation harness with comparison methods

Tests go in `tests/`. Run with `python -m pytest tests/ -v`.

## Active task
`.ml-metaopt/state.json` exists. Inspect its `status`.
If `status` is `RUNNING`, invoke the `ml-metaoptimization` skill immediately,
resume from `machine_state`, and do not ask the user for campaign inputs.
If `status` is terminal, remove this block, stop auto-resuming, and surface `next_action` to the operator without executing from it.
