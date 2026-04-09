# dg-bidmc — Infrastructure and Session Context

## Project

DG-GAT for respiratory boundary detection on BIDMC PhysioNet dataset.
See `AGENTS.md` for full technical context. See `RESEARCH_PLAN.md` for plan.

## Execution Environment

### Cluster Topology

```
Laptop (control plane — code edits, light tests, data loading)
    │
    ▼
ray-head @ 89.167.79.237 / 100.105.11.68 (Tailscale)
    │  hel1, Hetzner cx53, 16 CPU
    │  Ray 2.40.0, head at :6379
    │  Project synced to: /root/dg_bidmc/
    │  ray-hetzner repo at: /root/ray-hetzner/
    │
    └── Tailscale ──────────→ Aorus @ 100.104.191.44
                               16 CPU, 1 GPU (AMD RX 5700 XT, ROCm)
                               resource label: {"aorus": 1}
```

### Aorus GPU Worker — CRITICAL CONSTRAINTS

**NEVER pip-install or upgrade torch on Aorus.** Aorus runs a custom AMD ROCm build:
- `torch 2.9.1a0+gitd38164a` (system Python at `/usr/bin/python3`)
- Ray venv: `~/ray-venv` (Ray only — NO torch in venv)

To target Aorus from Ray tasks:
```python
@ray.remote(num_gpus=1, resources={"aorus": 1})  # GPU work on Aorus
def train_model(): ...

@ray.remote(resources={"aorus": 1})  # CPU-only work on Aorus
def cpu_task(): ...
```

### Re-attaching Aorus After Cluster Restart

The Aorus Ray worker is ephemeral — it dies on reboot. After any cluster restart:
```bash
cd ~/projects/ray-hetzner
./attach_aorus.sh
```

## Key Paths

| Location | Path |
|----------|------|
| Project (laptop) | `~/projects/dg_bidmc/` |
| Project (head) | `/root/dg_bidmc/` |
| ray-hetzner repo (laptop) | `~/projects/ray-hetzner/` |
| Constituency benchmark | `~/projects/dg_constituency_benchmark/` |
| DegreeGraph2 source | `~/projects/DegreeGraph2/` |
| Motifs source | `~/projects/Motifs/` |
| Cluster docs | `~/projects/ray-hetzner/CLAUDE.md` |

## Pushing Code to Head

```bash
cd ~/projects/ray-hetzner
./push_code.sh ~/projects/dg_bidmc /root/dg_bidmc
```

## External Libraries (NOT in requirements.txt)

Installed system-wide on Aorus and laptop:

- `degreegraph` — `compute_arrays(highs, lows)` → CSR arrays `(indices[int32], is_lows[int8], offsets[int32], connections[int32])`
  - Source: `/home/jakub/projects/DegreeGraph2` (editable install)
- `motifs` — `EdgeGraph.from_degreegraph2(indices, is_lows, offsets, connections, series)`
  - Forward edges only: `src_bar < dst_bar`
  - `edge_size` is `float32` — cast to `float64` before log-transform

## Running Tests Locally

```bash
cd ~/projects/dg_bidmc
python -m pytest tests/ -v
```

## GitHub Repository

https://github.com/jc1122/dg-bidmc

## Related Repositories

| Repo | GitHub | Purpose |
|------|--------|---------|
| dg-constituency-benchmark | jc1122/dg-constituency-benchmark | Origin project; DG-v5 pipeline |
| DegreeGraph2 | jc1122/DegreeGraph2 | Core graph library |
| Motifs | jc1122/Motifs | EdgeGraph API |
