#!/usr/bin/env python3
"""DegreeGraph2/Motifs adapter, level assignment, and parse tree recovery."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_method_result


def build_graph(signal: np.ndarray):
    """Build EdgeGraph from a raw signal.

    Args:
        signal: 1D numeric array (will be cast to float64)

    Returns:
        motifs.EdgeGraph instance
    """
    import degreegraph
    from motifs import EdgeGraph

    x64 = np.ascontiguousarray(signal, dtype=np.float64)
    indices, is_lows, offsets, connections = degreegraph.compute_arrays(x64, x64)
    graph = EdgeGraph.from_degreegraph2(indices, is_lows, offsets, connections, x64)
    return graph


def assign_levels(graph, n_levels: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Assign hierarchy levels to nodes using KMeans on log(edge_size).

    Level 0 = coarsest (phrase boundaries), n_levels-1 = finest (oscillations).

    Returns:
        (node_level, edge_level) — both int arrays
    """
    from sklearn.cluster import KMeans

    es = np.asarray(graph.edge_size, dtype=np.float64)

    if len(es) < n_levels:
        # Degenerate: not enough edges for clustering
        node_level = np.full(len(graph.node_bar), n_levels - 1, dtype=np.int32)
        edge_level = np.full(len(es), n_levels - 1, dtype=np.int32)
        return node_level, edge_level

    log_es = np.log(es + 1e-8).reshape(-1, 1)

    # Guard: if all edges have the same size, clustering is meaningless
    if log_es.std() < 1e-6:
        node_level = np.full(len(graph.node_bar), n_levels - 1, dtype=np.int32)
        edge_level = np.full(len(es), n_levels - 1, dtype=np.int32)
        return node_level, edge_level

    km = KMeans(n_clusters=n_levels, n_init=10, random_state=42)
    edge_level = km.fit_predict(log_es)

    # Sort clusters: largest edge_size = level 0 (coarsest)
    cluster_means = np.array([log_es[edge_level == k].mean() for k in range(n_levels)])
    rank = np.argsort(cluster_means)[::-1]  # descending
    remap = {int(old): int(new) for new, old in enumerate(rank)}
    edge_level = np.array([remap[int(l)] for l in edge_level], dtype=np.int32)

    # Assign each node: level = minimum (coarsest) level of any connected edge
    node_level = np.full(len(graph.node_bar), n_levels - 1, dtype=np.int32)
    src_rows = np.asarray(graph.edge_source_rows)
    dst_rows = np.asarray(graph.edge_to)

    for e in range(len(es)):
        src = int(src_rows[e])
        tgt = int(dst_rows[e])
        lev = int(edge_level[e])
        if lev < node_level[src]:
            node_level[src] = lev
        if lev < node_level[tgt]:
            node_level[tgt] = lev

    return node_level, edge_level


def recover_parse_tree(graph, node_level: np.ndarray, signal: np.ndarray) -> dict:
    """Reconstruct parse tree from DG node hierarchy.

    DG nodes sit at signal extrema (peaks/troughs). Terminal boundaries lie
    at zero-crossings BETWEEN consecutive opposite-sign extrema. The algorithm:

      1. Select structural nodes (level ≤ 1) as the major turning points.
      2. Enforce alternating peak/trough order, keeping the most extreme
         of any same-type run.
      3. Find the zero-crossing between each consecutive peak↔trough pair.
      4. Bookend with 0 and len(signal) to get complete boundaries.
      5. Label each terminal by sign of signal at its midpoint.
      6. Classify sentence type by terminal count (tolerance range).

    DG's contribution: in noisy signals, there are many spurious zero-crossings.
    DG's structural nodes identify which extrema pairs correspond to real
    terminal transitions, letting us select the correct zero-crossings.

    Args:
        graph: EdgeGraph with node_bar, node_is_low, edge_size, etc.
        node_level: int array of hierarchy levels per node
        signal: 1D numeric array of the original signal

    Returns dict with standard benchmark keys.
    """
    from common import D_MIN

    sig = np.asarray(signal, dtype=np.float64)
    bars = np.asarray(graph.node_bar)
    is_low = np.asarray(graph.node_is_low)
    es = np.asarray(graph.edge_size, dtype=np.float64)
    src_rows = np.asarray(graph.edge_source_rows)
    dst_rows = np.asarray(graph.edge_to)

    # Per-node significance = max edge_size of any connected edge
    node_significance = np.zeros(len(bars))
    for e in range(len(es)):
        src, tgt = int(src_rows[e]), int(dst_rows[e])
        if es[e] > node_significance[src]:
            node_significance[src] = es[e]
        if es[e] > node_significance[tgt]:
            node_significance[tgt] = es[e]

    sort_idx = np.argsort(bars)
    bars_sorted = bars[sort_idx]
    levels_sorted = node_level[sort_idx]
    is_low_sorted = is_low[sort_idx]
    significance_sorted = node_significance[sort_idx]

    # ── Structural nodes: level ≤ 1 (coarse + mid) ──────────────────────
    structural_mask = levels_sorted <= 1
    struct_bars = bars_sorted[structural_mask]
    struct_is_low = is_low_sorted[structural_mask]
    struct_sig_vals = np.array([sig[min(int(b), len(sig) - 1)]
                                for b in struct_bars])

    # ── Enforce alternating peaks/troughs ────────────────────────────────
    # When consecutive structural nodes have the same type, keep the most
    # extreme (lowest trough or highest peak).
    if len(struct_bars) > 0:
        filt_bars = [int(struct_bars[0])]
        filt_is_low = [bool(struct_is_low[0])]
        filt_vals = [struct_sig_vals[0]]

        for i in range(1, len(struct_bars)):
            bar_i = int(struct_bars[i])
            low_i = bool(struct_is_low[i])
            val_i = struct_sig_vals[i]

            if low_i != filt_is_low[-1]:
                # Alternation — accept
                filt_bars.append(bar_i)
                filt_is_low.append(low_i)
                filt_vals.append(val_i)
            else:
                # Same type — keep the more extreme
                if low_i:  # both troughs → keep lower
                    if val_i < filt_vals[-1]:
                        filt_bars[-1] = bar_i
                        filt_vals[-1] = val_i
                else:  # both peaks → keep higher
                    if val_i > filt_vals[-1]:
                        filt_bars[-1] = bar_i
                        filt_vals[-1] = val_i
    else:
        filt_bars, filt_is_low = [], []

    # ── Find zero-crossings between consecutive extrema pairs ────────────
    boundary_positions = [0]  # bookend: signal start
    for i in range(len(filt_bars) - 1):
        b1, b2 = filt_bars[i], filt_bars[i + 1]
        seg = sig[b1:b2 + 1]
        signs = np.sign(seg)
        nonzero_mask = signs != 0
        nonzero_signs = signs[nonzero_mask]
        nonzero_idx = np.where(nonzero_mask)[0]

        if len(nonzero_signs) > 1:
            flips = np.where(np.diff(nonzero_signs) != 0)[0]
            if len(flips) > 0:
                # Take the zero-crossing closest to the midpoint between nodes
                crossings = [b1 + (int(nonzero_idx[f]) + int(nonzero_idx[f + 1])) // 2
                             for f in flips]
                midpoint = (b1 + b2) / 2
                best = min(crossings, key=lambda c: abs(c - midpoint))
                boundary_positions.append(best)
    boundary_positions.append(len(sig))  # bookend: signal end

    # Deduplicate and sort
    boundaries = sorted(set(boundary_positions))

    # Build terminal pairs (skip zero-width)
    terminal_boundaries = [(boundaries[j], boundaries[j + 1])
                           for j in range(len(boundaries) - 1)
                           if boundaries[j] < boundaries[j + 1]]

    # ── Merge micro-segments (< D_MIN/4 bars) ───────────────────────────
    MIN_LEN = D_MIN // 4
    changed = True
    while changed:
        changed = False
        for i in range(len(terminal_boundaries)):
            seg_len = terminal_boundaries[i][1] - terminal_boundaries[i][0]
            if seg_len < MIN_LEN and len(terminal_boundaries) > 1:
                if i == 0:
                    merge_with = 1
                elif i == len(terminal_boundaries) - 1:
                    merge_with = i - 1
                else:
                    l_len = terminal_boundaries[i - 1][1] - terminal_boundaries[i - 1][0]
                    r_len = terminal_boundaries[i + 1][1] - terminal_boundaries[i + 1][0]
                    merge_with = i - 1 if l_len >= r_len else i + 1
                lo, hi = min(i, merge_with), max(i, merge_with)
                merged = (terminal_boundaries[lo][0], terminal_boundaries[hi][1])
                terminal_boundaries = terminal_boundaries[:lo] + [merged] + terminal_boundaries[hi + 1:]
                changed = True
                break

    # ── Labels from signal sign at midpoint ──────────────────────────────
    terminal_labels = []
    for start, end in terminal_boundaries:
        mid = min((start + end) // 2, len(sig) - 1)
        terminal_labels.append("Up" if sig[mid] > 0 else "Down")

    # ── Phrase boundaries from level-0 nodes ─────────────────────────────
    coarse_mask = levels_sorted == 0
    coarse_bars = bars_sorted[coarse_mask]
    phrase_boundaries = [(int(coarse_bars[i]), int(coarse_bars[i + 1]))
                         for i in range(len(coarse_bars) - 1)]

    # ── Sentence classification (tolerance) ──────────────────────────────
    n_terminals = len(terminal_labels)
    if n_terminals > 5:
        sentence_type = "impulse"
    elif n_terminals >= 2:
        sentence_type = "correction"
    else:
        sentence_type = "unknown"

    return {
        "terminal_boundaries_pred": terminal_boundaries,
        "terminal_labels_pred": terminal_labels,
        "phrase_boundaries_pred": phrase_boundaries,
        "sentence_type_pred": sentence_type,
        "diagnostics": {
            "n_nodes": len(bars),
            "n_coarse": int(coarse_mask.sum()),
            "n_mid": int(structural_mask.sum()),
            "n_fine": int((levels_sorted == 2).sum()),
            "n_structural_filtered": len(filt_bars),
            "n_terminals_pred": n_terminals,
            "signal_length": len(sig),
        },
    }


def run_dg_pipeline(signal: np.ndarray) -> dict:
    """Full DG pipeline: build graph → assign levels → recover parse tree.

    Returns the parse tree dict from recover_parse_tree().
    """
    graph = build_graph(signal)
    node_level, edge_level = assign_levels(graph, n_levels=3)
    result = recover_parse_tree(graph, node_level, signal)
    result["diagnostics"]["edge_level_counts"] = {
        int(k): int(v) for k, v in zip(*np.unique(edge_level, return_counts=True))
    }
    return result


def run_dg_benchmark(signals, labels, terminal_starts_list, terminal_ends_list,
                     terminal_labels_list, **kwargs):
    """Run the DG pipeline on a full dataset and return a method result.

    This is the benchmark interface used by scripts/benchmark.py.
    """
    from common import timer

    sentence_preds = []
    boundary_preds = []
    label_preds = []

    with timer() as t:
        for i, signal in enumerate(signals):
            result = run_dg_pipeline(signal)
            sentence_preds.append(result["sentence_type_pred"])

            # Extract predicted starts/ends for metric computation
            if result["terminal_boundaries_pred"]:
                starts = [b[0] for b in result["terminal_boundaries_pred"]]
                ends = [b[1] for b in result["terminal_boundaries_pred"]]
            else:
                starts, ends = [], []

            boundary_preds.append({
                "starts": np.array(starts, dtype=np.int32),
                "ends": np.array(ends, dtype=np.int32),
            })
            label_preds.append(result["terminal_labels_pred"])

    return make_method_result(
        name="DegreeGraph",
        sentence_preds=sentence_preds,
        boundary_preds=boundary_preds,
        label_preds=label_preds,
        timing_sec=t.elapsed,
    )
