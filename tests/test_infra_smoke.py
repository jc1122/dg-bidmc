"""
Infrastructure smoke tests — verify Aorus GPU worker is reachable via Ray
and all project dependencies are importable and functional on that worker.

Run from the head node (same Ray/Python version as cluster):
    python3 tests/test_infra_smoke.py
    pytest tests/test_infra_smoke.py -v --timeout=120

Override address via env var (e.g. "auto" when on the head):
    RAY_ADDRESS=auto python3 tests/test_infra_smoke.py
"""

from __future__ import annotations

import os
import sys

import ray

# "auto" → connect to local cluster when running on the head node.
# Override via RAY_ADDRESS env var if needed.
RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "auto")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def init_ray():
    if not ray.is_initialized():
        # working_dir ships scripts/ from head → all workers (incl. Aorus),
        # so `from dg_pipeline import ...` works without the project on Aorus.
        ray.init(RAY_ADDRESS, ignore_reinit_error=True,
                 runtime_env={"working_dir": "/root/dg_bidmc/scripts"})


# ---------------------------------------------------------------------------
# Remote tasks — each pinned to Aorus
# ---------------------------------------------------------------------------


# max_calls removed — GPU visibility is guaranteed by num_gpus=1; max_calls=1 caused
# the worker-exit HIP death notification to race with result delivery on the head
# (head has no torch → can't deserialize torch-typed exit metadata).
@ray.remote(num_gpus=1, resources={"aorus": 1})
def task_torch_gpu() -> dict:
    """Verify custom ROCm torch build and GPU compute on Aorus."""
    try:
        import torch

        assert torch.cuda.is_available(), "CUDA/ROCm not available"
        device_name = torch.cuda.get_device_name(0)

        # Round-trip GPU matmul
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        c = (a @ b).cpu()
        assert c.shape == (256, 256)

        return {"torch_version": str(torch.__version__), "gpu": device_name, "matmul_ok": True}
    except Exception as exc:
        # Wrap so head (which has no torch) can always deserialise the result
        return {"matmul_ok": False, "error": str(exc)}


@ray.remote(resources={"aorus": 1})
def task_torch_geometric() -> dict:
    """Verify torch_geometric import and basic graph forward pass on GPU."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Minimal graph: 5 nodes, 4 edges
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    x = torch.randn(5, 6)
    edge_attr = torch.randn(4, 2)

    conv = GATConv(6, 16, heads=2, edge_dim=2).to(device)
    out = conv(x.to(device), edge_index.to(device), edge_attr.to(device))
    assert out.shape == (5, 32), f"unexpected shape: {out.shape}"

    import torch_geometric
    return {"torch_geometric_version": torch_geometric.__version__, "gat_ok": True}


_SCRIPTS_RUNTIME = {}  # kept for compat; working_dir set at job level in init_ray()


@ray.remote(resources={"aorus": 1})
def task_degreegraph() -> dict:
    """Verify degreegraph + motifs on a synthetic respiratory-like signal."""
    import numpy as np
    from scipy.signal import butter, filtfilt
    import degreegraph
    from motifs import EdgeGraph

    # Synthetic 8-second sine wave at 125 Hz (mimics slow breathing ~1 Hz)
    fs = 125
    t = np.linspace(0, 8, 8 * fs, endpoint=False)
    signal = np.sin(2 * np.pi * 0.8 * t) + 0.05 * np.random.default_rng(0).standard_normal(len(t))

    # LP filter as per project spec
    b, a = butter(4, 2.0 / (fs / 2), "low")
    sig_lp = filtfilt(b, a, signal - signal.mean())

    sig64 = np.ascontiguousarray(sig_lp, dtype=np.float64)
    indices, is_lows, offsets, connections = degreegraph.compute_arrays(sig64, sig64)
    graph = EdgeGraph.from_degreegraph2(indices, is_lows, offsets, connections, sig64)

    n_nodes = len(graph.node_bar)
    n_edges = len(graph.edge_size)
    assert n_nodes > 0, "graph has no nodes"
    assert n_edges > 0, "graph has no edges"

    # Cast float32 edge_size to float64 before log (project-critical!)
    es = np.asarray(graph.edge_size, dtype=np.float64)
    log_es = np.log(es + 1e-8)
    assert np.all(np.isfinite(log_es)), "log_edge_size contains non-finite values"

    return {"n_nodes": n_nodes, "n_edges": n_edges, "dg_ok": True}


@ray.remote(resources={"aorus": 1})
def task_dg_pipeline() -> dict:
    """Verify build_graph + assign_levels from project scripts on Aorus."""
    import numpy as np
    from scipy.signal import butter, filtfilt
    from dg_pipeline import build_graph, assign_levels

    fs = 125
    t = np.linspace(0, 8, 8 * fs, endpoint=False)
    signal = np.sin(2 * np.pi * 0.8 * t)
    b, a = butter(4, 2.0 / (fs / 2), "low")
    sig_lp = filtfilt(b, a, signal - signal.mean())

    graph = build_graph(sig_lp)
    node_level, edge_level = assign_levels(graph, n_levels=3)

    assert len(node_level) == len(graph.node_bar)
    assert set(node_level).issubset({0, 1, 2})

    return {
        "n_nodes": len(graph.node_bar),
        "levels": sorted(set(node_level.tolist())),
        "pipeline_ok": True,
    }


@ray.remote(num_gpus=1, resources={"aorus": 1})
def task_full_dggat_forward() -> dict:
    """Instantiate DGGAT stub and do a GPU forward pass with random data.

    NOTE: All GATConv layers use add_self_loops=False.  DG graphs are DAGs
    with no self-loops; the default add_self_loops=True internally calls
    remove_self_loops via boolean tensor indexing, which triggers
    hipErrorInvalidDeviceFunction on gfx1010 (RX 5700 XT) due to a missing
    compiled kernel.  add_self_loops=False bypasses that code path entirely.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv

        device = "cuda"

        # Mimic DGGAT architecture from AGENTS.md
        class _DGGAT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                in_dim, hidden, heads = 6, 64, 8
                # add_self_loops=False avoids broken nonzero kernel on gfx1010
                kw = dict(edge_dim=2, add_self_loops=False)
                self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=True, **kw)
                self.conv2 = GATConv(hidden * heads, hidden, heads=heads, concat=True, **kw)
                self.conv3 = GATConv(hidden * heads, hidden, heads=1, concat=False, **kw)
                self.boundary_head = torch.nn.Linear(hidden, 1)

            def forward(self, x, edge_index, edge_attr):
                x = F.elu(self.conv1(x, edge_index, edge_attr))
                x = F.elu(self.conv2(x, edge_index, edge_attr))
                x = self.conv3(x, edge_index, edge_attr)
                return self.boundary_head(x).squeeze(-1)

        model = _DGGAT().to(device)

        n_nodes, n_edges = 20, 30
        x = torch.randn(n_nodes, 6, device=device)
        edge_index = torch.randint(0, n_nodes, (2, n_edges), device=device)
        edge_attr = torch.randn(n_edges, 2, device=device)

        logits = model(x, edge_index, edge_attr)
        assert logits.shape == (n_nodes,), f"bad shape: {logits.shape}"

        total_params = sum(p.numel() for p in model.parameters())
        return {
            "logits_shape": list(logits.shape),
            "total_params": total_params,
            "device": str(logits.device),
            "dggat_gpu_ok": True,
        }
    except Exception as exc:
        return {"dggat_gpu_ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_all() -> bool:
    init_ray()
    print(f"\nRay cluster resources: {ray.cluster_resources()}\n")

    tests = [
        ("torch_gpu",          task_torch_gpu.remote()),
        ("torch_geometric",    task_torch_geometric.remote()),
        ("degreegraph+motifs", task_degreegraph.remote()),
        ("dg_pipeline",        task_dg_pipeline.remote()),
        ("dggat_gpu_forward",  task_full_dggat_forward.remote()),
    ]

    passed = failed = 0
    for name, ref in tests:
        try:
            result = ray.get(ref, timeout=120)
            print(f"  [PASS] {name}: {result}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------

import pytest  # noqa: E402 (after stdlib imports)


@pytest.fixture(scope="module", autouse=True)
def ray_session():
    init_ray()
    yield
    # Don't shut down — cluster stays alive for the session


def test_torch_gpu():
    result = ray.get(task_torch_gpu.remote(), timeout=60)
    assert result["matmul_ok"]
    assert "5700 XT" in result["gpu"] or "AMD" in result["gpu"]


def test_torch_geometric():
    result = ray.get(task_torch_geometric.remote(), timeout=60)
    assert result["gat_ok"]


def test_degreegraph():
    result = ray.get(task_degreegraph.remote(), timeout=60)
    assert result["dg_ok"]
    assert result["n_nodes"] > 5
    assert result["n_edges"] > 5


def test_dg_pipeline():
    result = ray.get(task_dg_pipeline.remote(), timeout=60)
    assert result["pipeline_ok"]
    assert 0 in result["levels"]


def test_dggat_gpu_forward():
    result = ray.get(task_full_dggat_forward.remote(), timeout=120)
    assert result["dggat_gpu_ok"]
    assert "cuda" in result["device"]
    assert result["logits_shape"] == [20]


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
