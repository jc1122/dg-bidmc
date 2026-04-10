"""Training loop for DG-GNN respiratory boundary detection.

Reads a config dict specifying the full experiment configuration:
graph construction, features, architecture, training hyperparameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import build_model, count_parameters
from graph_features import compute_feature_dims, cache_split_graphs


def _apply_soft_labels(batch, sigma: float, tol: int) -> torch.Tensor:
    """Convert hard binary labels to Gaussian soft labels on-the-fly.

    Uses hard labels (y==1) to locate GT boundary nodes, then computes
    Gaussian-weighted scores for all nodes within *tol* samples.
    Runs on CPU to avoid gfx1010 missing-kernel issues with boolean indexing.
    """
    bars = batch.node_bars.cpu().float()
    y_hard = batch.y.cpu()
    gt_indices = (y_hard >= 0.5).nonzero(as_tuple=True)[0]
    if len(gt_indices) == 0:
        return batch.y

    gt_bars = bars[gt_indices]
    dists = torch.abs(bars.unsqueeze(1) - gt_bars.unsqueeze(0))
    weights = torch.exp(-0.5 * (dists / sigma) ** 2)
    weights[dists > tol] = 0.0
    y_soft, _ = weights.max(dim=1)
    return y_soft.to(batch.y.device)


def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, pos_weight=None):
    """Focal loss for binary classification (operates on logits).
    
    Reduces relative loss for well-classified examples, focusing training
    on hard negatives/positives. Compatible with class imbalance via pos_weight.
    """
    if pos_weight is not None:
        weight = torch.where(targets >= 0.5, pos_weight, torch.ones_like(targets))
    else:
        weight = torch.ones_like(targets)
    
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = torch.where(targets >= 0.5, p, 1 - p)
    focal_weight = alpha * (1 - pt) ** gamma
    loss = (focal_weight * weight * bce).mean()
    return loss


def load_cached_graphs(data_dir: str) -> list:
    """Load all .pt files from a directory."""
    from torch_geometric.data import Data
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.pt"))
    return [torch.load(f, weights_only=False) for f in files]


def _rebuild_graphs_if_needed(
    data_dir: str, split_name: str, config: dict, verbose: bool = True
) -> str:
    """Rebuild cached graphs when feature dims don't match config.

    Returns the (possibly updated) data_dir path.
    """
    expected_in, expected_edge = compute_feature_dims(config)
    existing = sorted(Path(data_dir).glob("*.pt"))
    if existing:
        sample = torch.load(existing[0], weights_only=False)
        cached_in = sample.x.shape[1] if sample.x.dim() == 2 else 0
        cached_edge = sample.edge_attr.shape[1] if sample.edge_attr.dim() == 2 else 0
        if cached_in == expected_in and cached_edge == expected_edge:
            # Check if augmentation config matches via marker file
            n_aug = config.get('n_augmented_copies', 0) if config else 0
            marker = Path(data_dir) / '.augment_marker'
            if split_name == 'train' and n_aug > 0:
                cached_n_aug = 0
                if marker.exists():
                    try:
                        cached_n_aug = int(marker.read_text().strip())
                    except (ValueError, OSError):
                        pass
                if cached_n_aug != n_aug:
                    if verbose:
                        print(f"  Augmentation rebuild: cached n_aug={cached_n_aug}, need={n_aug}")
                    cached_in = -1  # force rebuild
                else:
                    return data_dir
            else:
                return data_dir  # dims match, use as-is
    else:
        cached_in, cached_edge = 0, 0

    if verbose:
        print(f"  Rebuilding {split_name} graphs: cached=({cached_in},{cached_edge}), "
              f"need=({expected_in},{expected_edge})")

    from data_loader import load_all_patients, get_splits, profile_patient
    patients = load_all_patients(verbose=False)
    splits = get_splits()

    # Determine which patient IDs to use
    if split_name == "val_adversarial":
        split_ids = [
            pid for pid in splits["val"]
            if profile_patient(patients[pid]).get("is_adversarial", False)
        ]
    elif split_name in splits:
        split_ids = splits[split_name]
    else:
        # Infer from existing filenames
        split_ids = sorted(set(
            f.stem.rsplit("_w", 1)[0] for f in Path(data_dir).glob("*.pt")
        ))

    wb = config.get("window_breaths", 6)
    # Clear old files
    for f in Path(data_dir).glob("*.pt"):
        f.unlink()
    cache_split_graphs(patients, split_ids, data_dir, config=config,
                       window_breaths=wb)
    if verbose:
        n_new = len(list(Path(data_dir).glob("*.pt")))
        print(f"  Rebuilt {n_new} graphs for {split_name}")
    return data_dir

def compute_boundary_f1(pred_troughs, gt_troughs, tol_samples=75):
    """Compute boundary F1 at ±tol_samples tolerance.
    
    Uses greedy closest-first matching (same as common.py boundary_f1).
    """
    if len(pred_troughs) == 0 or len(gt_troughs) == 0:
        return 0.0
    
    pairs = []
    for i, gt in enumerate(gt_troughs):
        for j, pred in enumerate(pred_troughs):
            dist = abs(float(pred) - float(gt))
            if dist <= tol_samples:
                pairs.append((dist, i, j))
    pairs.sort()
    used_gt, used_pred = set(), set()
    matched = 0
    for _, i, j in pairs:
        if i not in used_gt and j not in used_pred:
            matched += 1
            used_gt.add(i)
            used_pred.add(j)
    
    precision = matched / len(pred_troughs) if len(pred_troughs) > 0 else 0
    recall = matched / len(gt_troughs) if len(gt_troughs) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_rate_mae(pred_troughs, gt_troughs, fs=125):
    """Compute respiratory rate MAE in bpm from trough positions."""
    if len(pred_troughs) < 2 or len(gt_troughs) < 2:
        return float('inf')
    pred_intervals = np.diff(sorted(pred_troughs)) / fs
    gt_intervals = np.diff(sorted(gt_troughs)) / fs
    pred_rate = 60.0 / np.mean(pred_intervals) if np.mean(pred_intervals) > 0 else 0
    gt_rate = 60.0 / np.mean(gt_intervals) if np.mean(gt_intervals) > 0 else 0
    return abs(pred_rate - gt_rate)

def train_one_epoch(model, loader, optimizer, config, device):
    """Train for one epoch. Returns (mean_loss, device_used).
    
    Robust to GPU crashes: if a HIP/CUDA error occurs mid-batch,
    moves model+data to CPU and retries remaining batches.
    """
    model.train()
    total_loss = 0
    n_graphs = 0
    
    pw_val = config.get('pos_weight', 7.0)
    pos_weight = torch.tensor(pw_val, device=device)
    rate_w = config.get('loss_rate_weight', 0.0)
    type_w = config.get('loss_type_weight', 0.0)
    use_focal = config.get('boundary_loss', 'bce') == 'focal'
    focal_gamma = config.get('focal_gamma', 2.0)
    focal_alpha = config.get('focal_alpha', 0.25)
    label_sigma = config.get('label_sigma', 0)
    tol_samples = config.get('tol_samples_train', 75)
    
    for batch in loader:
        try:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, 
                         edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                         batch=batch.batch if hasattr(batch, 'batch') else None)
            
            # Compute soft labels on-the-fly if label_sigma > 0
            targets = batch.y
            if label_sigma > 0 and hasattr(batch, 'node_bars'):
                targets = _apply_soft_labels(batch, label_sigma, tol_samples)
            
            # Boundary loss (main)
            if use_focal:
                bce = focal_loss_with_logits(
                    out['boundary_logits'], targets,
                    alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
            else:
                bce = F.binary_cross_entropy_with_logits(
                    out['boundary_logits'], targets, pos_weight=pos_weight)
            
            loss = bce
            
            # Rate aux loss
            if rate_w > 0 and 'rate_pred' in out and hasattr(batch, 'rate_target'):
                rate_loss = F.mse_loss(out['rate_pred'].squeeze(), batch.rate_target)
                loss = loss + rate_w * rate_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs
        except RuntimeError as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ('hip', 'cuda', 'device-side assert', 'invalid device function', 'out of memory')):
                print(f"  [GPU ERROR] {e}")
                if device != 'cpu':
                    print("  Falling back to CPU for remainder of epoch")
                    device = 'cpu'
                    model.cpu()
                    pos_weight = pos_weight.cpu()
                    for pg in optimizer.param_groups:
                        for p in pg['params']:
                            state = optimizer.state.get(p, {})
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.cpu()
                    # Retry this batch on CPU
                    batch = batch.to('cpu')
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index,
                                 edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                                 batch=batch.batch if hasattr(batch, 'batch') else None)
                    # Compute soft labels for CPU fallback
                    cpu_targets = batch.y
                    if label_sigma > 0 and hasattr(batch, 'node_bars'):
                        cpu_targets = _apply_soft_labels(batch, label_sigma, tol_samples)
                    if use_focal:
                        bce = focal_loss_with_logits(
                            out['boundary_logits'], cpu_targets,
                            alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
                    else:
                        bce = F.binary_cross_entropy_with_logits(
                            out['boundary_logits'], cpu_targets, pos_weight=pos_weight)
                    loss = bce
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * batch.num_graphs
                    n_graphs += batch.num_graphs
                else:
                    raise
            else:
                raise
    
    return total_loss / max(n_graphs, 1), device

@torch.no_grad()
def nms_1d(bars, scores, min_dist):
    """Non-max suppression: keep highest-score node within each min_dist window."""
    if len(bars) == 0:
        return bars
    order = np.argsort(-scores)
    keep = []
    suppressed = set()
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for other in order:
            if other != idx and abs(int(bars[other]) - int(bars[idx])) < min_dist:
                suppressed.add(other)
    return bars[sorted(keep)]


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, tol_samples=75, fs=125,
             nms_dist=0, top_n=0):
    """Evaluate model on a data loader. Returns dict with metrics.
    
    Args:
        top_n: if >0, after NMS keep only top-N predictions by score per graph.
               This eliminates FP from over-prediction.
    """
    model.eval()
    all_f1s = []
    all_rate_maes = []
    total_loss = 0
    n_graphs = 0
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index,
                     edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                     batch=batch.batch if hasattr(batch, 'batch') else None)
        
        # Loss
        bce = F.binary_cross_entropy_with_logits(out['boundary_logits'], batch.y)
        total_loss += bce.item() * batch.num_graphs
        n_graphs += batch.num_graphs
        
        # Per-graph F1
        logits = out['boundary_logits'].cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))  # sigmoid
        labels = batch.y.cpu().numpy()
        node_bars = batch.node_bars.cpu().numpy()
        batch_ids = batch.batch.cpu().numpy() if hasattr(batch, 'batch') and batch.batch is not None else np.zeros(len(logits), dtype=int)
        
        for g in range(batch.num_graphs):
            mask = batch_ids == g
            g_scores = scores[mask]
            g_bars = node_bars[mask]
            g_labels = labels[mask]
            
            above = g_scores > threshold
            if nms_dist > 0 and above.sum() > 0:
                pred_troughs = nms_1d(g_bars[above], g_scores[above], nms_dist)
            else:
                pred_troughs = g_bars[above]
            
            # Top-N constraint: keep only highest-scored predictions
            if top_n > 0 and len(pred_troughs) > top_n:
                pred_scores = np.array([
                    g_scores[np.argmin(np.abs(g_bars - pb))]
                    for pb in pred_troughs
                ])
                keep_idx = np.argsort(-pred_scores)[:top_n]
                pred_troughs = np.sort(pred_troughs[keep_idx])
            
            gt_troughs = g_bars[g_labels > 0.5]
            
            f1 = compute_boundary_f1(pred_troughs, gt_troughs, tol_samples)
            all_f1s.append(f1)
            
            mae = compute_rate_mae(pred_troughs, gt_troughs, fs)
            if mae != float('inf'):
                all_rate_maes.append(mae)
    
    return {
        'boundary_f1_600ms': float(np.mean(all_f1s)) if all_f1s else 0.0,
        'rate_mae_bpm': float(np.mean(all_rate_maes)) if all_rate_maes else float('inf'),
        'loss': total_loss / max(n_graphs, 1),
        'n_graphs': n_graphs,
    }


def _collect_predictions(model, loader, device, mc_samples=1,
                         calibration_method='none', calibration_params=None,
                         include_features=False):
    """Run model on all data, return per-graph predictions.
    
    Args:
        mc_samples: Number of forward passes with dropout for MC dropout (1=standard)
        calibration_method: 'none', 'temperature', or 'platt'
        calibration_params: dict with 'temperature' (float) or 'platt_slope'+'platt_intercept'
        include_features: if True, each tuple has a 4th element: node features array
    
    Returns list of (scores, bars, labels[, features]) tuples, one per graph.
    """
    if mc_samples > 1:
        model.train()
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d,)):
                m.eval()
    else:
        model.eval()
    
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if mc_samples > 1:
                batch_logits = []
                for _ in range(mc_samples):
                    out = model(batch.x, batch.edge_index,
                                edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                                batch=batch.batch if hasattr(batch, 'batch') else None)
                    batch_logits.append(out['boundary_logits'].cpu().numpy())
                logits = np.mean(batch_logits, axis=0)
            else:
                out = model(batch.x, batch.edge_index,
                            edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                            batch=batch.batch if hasattr(batch, 'batch') else None)
                logits = out['boundary_logits'].cpu().numpy()
            
            # Apply calibration
            if calibration_method == 'temperature' and calibration_params:
                T = calibration_params.get('temperature', 1.0)
                logits = logits / T
            elif calibration_method == 'platt' and calibration_params:
                slope = calibration_params.get('platt_slope', 1.0)
                intercept = calibration_params.get('platt_intercept', 0.0)
                logits = slope * logits + intercept
            
            scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
            labels = batch.y.cpu().numpy()
            node_bars = batch.node_bars.cpu().numpy()
            feats = batch.x.cpu().numpy() if include_features else None
            batch_ids = batch.batch.cpu().numpy() if hasattr(batch, 'batch') and batch.batch is not None else np.zeros(len(logits), dtype=int)
            for g in range(batch.num_graphs):
                mask = batch_ids == g
                entry = (scores[mask], node_bars[mask], labels[mask])
                if include_features:
                    entry = entry + (feats[mask],)
                results.append(entry)
    
    model.eval()  # Restore
    return results


def _collect_raw_logits(model, loader, device, mc_samples=1):
    """Collect raw pre-sigmoid logits from model. For calibration fitting.
    
    If mc_samples > 1, runs multiple forward passes with dropout active
    and averages the logits.
    """
    if mc_samples > 1:
        model.train()
        # Keep BatchNorm in eval mode to avoid corrupting running stats
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d,)):
                m.eval()
    else:
        model.eval()
    
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if mc_samples > 1:
                # Multiple forward passes, average logits
                batch_logits = []
                for _ in range(mc_samples):
                    out = model(batch.x, batch.edge_index,
                                edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                                batch=batch.batch if hasattr(batch, 'batch') else None)
                    batch_logits.append(out['boundary_logits'].cpu().numpy())
                logits = np.mean(batch_logits, axis=0)
            else:
                out = model(batch.x, batch.edge_index,
                            edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                            batch=batch.batch if hasattr(batch, 'batch') else None)
                logits = out['boundary_logits'].cpu().numpy()
            all_logits.append(logits)
            all_labels.append(batch.y.cpu().numpy())
    
    model.eval()  # Restore eval mode
    return np.concatenate(all_logits), np.concatenate(all_labels)


def learn_temperature(logits, labels):
    """Learn temperature scaling parameter T to minimize NLL on validation logits.
    
    Returns optimal T in [0.1, 10.0].
    """
    def nll(T):
        scaled = logits / T
        # Numerically stable BCE: max(0, z) - z*y + log(1 + exp(-|z|))
        z = scaled
        loss = np.maximum(z, 0) - z * labels + np.log(1 + np.exp(-np.abs(z)))
        return loss.mean()
    
    # Grid search (robust, no scipy dependency)
    best_T = 1.0
    best_nll = nll(1.0)
    for T in np.arange(0.1, 10.05, 0.05):
        val = nll(T)
        if val < best_nll:
            best_nll = val
            best_T = T
    # Fine search around best
    for T in np.arange(max(0.1, best_T - 0.1), best_T + 0.11, 0.01):
        val = nll(T)
        if val < best_nll:
            best_nll = val
            best_T = T
    return float(best_T)


def fit_platt_scaling(logits, labels):
    """Fit Platt scaling (logistic regression) on validation logits.
    
    Returns (slope, intercept) tuple for: calibrated = sigmoid(slope * logit + intercept)
    """
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(solver='lbfgs', C=1e10, max_iter=10000,
                            class_weight='balanced')
    lr.fit(logits.reshape(-1, 1), (labels > 0.5).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _composite_nms_1d(bars, scores, duration, amplitude, min_dist,
                      alpha=1.0, beta=0.0, gamma=0.0):
    """NMS using composite ranking: alpha*score + beta*duration + gamma*(-amplitude).

    Nodes are suppressed based on composite rank order, but the returned
    array preserves the original GNN scores for downstream top-n selection.
    """
    if len(bars) == 0:
        return bars, scores
    composite = alpha * scores + beta * duration + gamma * (-amplitude)
    order = np.argsort(-composite)
    keep = []
    suppressed = set()
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for other in order:
            if other != idx and other not in suppressed:
                if abs(int(bars[other]) - int(bars[idx])) < min_dist:
                    suppressed.add(other)
    keep = sorted(keep)
    return bars[keep], scores[keep]


def _apply_post_processing(g_scores, g_bars, threshold, nms_dist, top_n,
                           adaptive_nms_frac=0.0, adaptive_top_n=False,
                           g_features=None, cnms_alpha=1.0, cnms_beta=0.0,
                           cnms_gamma=0.0):
    """Apply threshold → NMS → top_n pipeline for one graph.
    
    If adaptive_nms_frac > 0, NMS distance = max(nms_dist, frac × median_spacing)
    where median_spacing is estimated from above-threshold node positions.
    If adaptive_top_n is True, top_n is estimated from window duration and score peaks.
    If g_features is provided and cnms_beta/cnms_gamma > 0, uses composite NMS
    ranking with node duration (col 4) and amplitude (col 3) features.
    """
    above = g_scores > threshold
    if above.sum() == 0:
        return np.array([], dtype=g_bars.dtype)

    cand_bars = g_bars[above]
    cand_scores = g_scores[above]
    
    # Adaptive NMS: scale distance based on estimated breath spacing
    eff_nms = nms_dist
    if adaptive_nms_frac > 0 and len(cand_bars) >= 2:
        sorted_bars = np.sort(cand_bars)
        spacings = np.diff(sorted_bars)
        if len(spacings) > 0:
            median_spacing = np.median(spacings[spacings > 10])  # ignore tiny gaps
            if np.isfinite(median_spacing):
                eff_nms = max(nms_dist, int(adaptive_nms_frac * median_spacing))
    
    # Use composite NMS if features provided and weights are non-zero
    use_composite = (g_features is not None and (cnms_beta > 0 or cnms_gamma > 0))
    if eff_nms > 0:
        if use_composite:
            cand_dur = g_features[above, 4]   # duration_norm column
            cand_amp = g_features[above, 3]   # amplitude_norm column
            pred_troughs, pred_scores = _composite_nms_1d(
                cand_bars, cand_scores, cand_dur, cand_amp, eff_nms,
                alpha=cnms_alpha, beta=cnms_beta, gamma=cnms_gamma)
        else:
            pred_troughs = nms_1d(cand_bars, cand_scores, eff_nms)
            pred_scores = None
    else:
        pred_troughs = cand_bars
        pred_scores = None
    
    # Adaptive top_n: estimate expected boundary count from window span
    eff_top_n = top_n
    if adaptive_top_n and len(g_bars) >= 2:
        window_span = float(g_bars.max() - g_bars.min())
        if window_span > 0 and len(cand_bars) >= 2:
            sorted_cand = np.sort(cand_bars)
            spacings = np.diff(sorted_cand)
            valid_spacings = spacings[spacings > 20]
            if len(valid_spacings) >= 2:
                est_breath_period = np.median(valid_spacings)
                eff_top_n = max(2, int(np.round(window_span / est_breath_period)))
    
    if eff_top_n > 0 and len(pred_troughs) > eff_top_n:
        if pred_scores is not None:
            # Use original GNN scores for top-n selection after composite NMS
            keep_idx = np.argsort(-pred_scores)[:eff_top_n]
        else:
            _ps = np.array([
                g_scores[np.argmin(np.abs(g_bars - pb))]
                for pb in pred_troughs
            ])
            keep_idx = np.argsort(-_ps)[:eff_top_n]
        pred_troughs = np.sort(pred_troughs[keep_idx])
    
    return pred_troughs


def _eval_threshold(graph_preds, threshold, nms_dist, top_n, tol_samples, fs=125,
                    adaptive_nms_frac=0.0, adaptive_top_n=False,
                    cnms_alpha=1.0, cnms_beta=0.0, cnms_gamma=0.0):
    """Pure-numpy threshold evaluation on pre-collected predictions.
    
    graph_preds entries may be (scores, bars, labels) or
    (scores, bars, labels, features) when composite NMS is used.
    """
    all_f1s = []
    for entry in graph_preds:
        g_scores, g_bars, g_labels = entry[0], entry[1], entry[2]
        g_feats = entry[3] if len(entry) > 3 else None
        pred_troughs = _apply_post_processing(
            g_scores, g_bars, threshold, nms_dist, top_n,
            adaptive_nms_frac=adaptive_nms_frac,
            adaptive_top_n=adaptive_top_n,
            g_features=g_feats, cnms_alpha=cnms_alpha,
            cnms_beta=cnms_beta, cnms_gamma=cnms_gamma)
        gt_troughs = g_bars[g_labels > 0.5]
        all_f1s.append(compute_boundary_f1(pred_troughs, gt_troughs, tol_samples))
    return float(np.mean(all_f1s)) if all_f1s else 0.0


def optimize_threshold(model, loader, device, tol_samples=75, fs=125,
                       nms_dist=0, window_breaths=6, adaptive_nms=False,
                       mc_samples=1, calibration_method='none', calibration_params=None,
                       composite_nms=False):
    """Search for threshold+NMS+top_n that maximizes boundary F1.
    
    Collects model predictions once, then does fast numpy grid search.
    If adaptive_nms=True, also searches adaptive NMS fraction and adaptive top_n.
    If composite_nms=True, also searches composite NMS weights (alpha, beta, gamma)
    using node duration and amplitude features for ranking during NMS.
    """
    graph_preds = _collect_predictions(model, loader, device, mc_samples=mc_samples,
                                       calibration_method=calibration_method,
                                       calibration_params=calibration_params,
                                       include_features=composite_nms)
    
    best_f1 = 0.0
    best_thresh = 0.5
    best_nms = nms_dist
    best_top_n = 0
    best_anms_frac = 0.0
    best_atopn = False
    best_cnms = (1.0, 0.0, 0.0)  # (alpha, beta, gamma)
    
    # NMS fractions to try: 0 = off, 0.3-0.7 = fraction of median breath spacing
    anms_fracs = [0.0]
    atopn_opts = [False]
    if adaptive_nms:
        anms_fracs = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
        atopn_opts = [False, True]
    
    # Composite NMS weights to try
    cnms_configs = [(1.0, 0.0, 0.0)]  # baseline: standard NMS
    if composite_nms:
        for ca in [0.85, 0.9, 1.0]:
            for cb in [0.05, 0.1]:
                for cg in [0.1, 0.15, 0.2, 0.225]:
                    cnms_configs.append((ca, cb, cg))
    
    # Phase 1: coarse grid
    for thresh in np.arange(0.05, 0.90, 0.05):
        for nd in [0, 50, 75, 100, 120, 150, 200]:
            for tn in [0, window_breaths]:
                for af in anms_fracs:
                    for at in atopn_opts:
                        for ca, cb, cg in cnms_configs:
                            f1 = _eval_threshold(
                                graph_preds, thresh, nd, tn, tol_samples, fs,
                                adaptive_nms_frac=af, adaptive_top_n=at,
                                cnms_alpha=ca, cnms_beta=cb, cnms_gamma=cg)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_thresh = float(thresh)
                                best_nms = int(nd)
                                best_top_n = int(tn)
                                best_anms_frac = float(af)
                                best_atopn = bool(at)
                                best_cnms = (ca, cb, cg)
    
    # Phase 2: fine search around best
    fine_anms = [best_anms_frac]
    if adaptive_nms and best_anms_frac > 0:
        fine_anms = [max(0.1, best_anms_frac - 0.1),
                     best_anms_frac - 0.05,
                     best_anms_frac,
                     best_anms_frac + 0.05,
                     min(0.9, best_anms_frac + 0.1)]
    
    fine_cnms = [best_cnms]
    if composite_nms and (best_cnms[1] > 0 or best_cnms[2] > 0):
        ba, bb, bg = best_cnms
        for da in [-0.05, 0, 0.05]:
            for db in [-0.025, 0, 0.025]:
                for dg in [-0.025, 0, 0.025]:
                    c = (ba+da, max(0, bb+db), max(0, bg+dg))
                    if c not in fine_cnms:
                        fine_cnms.append(c)
    
    for thresh in np.arange(max(0.02, best_thresh - 0.10),
                            min(0.95, best_thresh + 0.10), 0.01):
        for nd in range(max(0, best_nms - 30), best_nms + 35, 5):
            for tn in [0, window_breaths]:
                for af in fine_anms:
                    for ca, cb, cg in fine_cnms:
                        f1 = _eval_threshold(
                            graph_preds, thresh, nd, tn, tol_samples, fs,
                            adaptive_nms_frac=af, adaptive_top_n=best_atopn,
                            cnms_alpha=ca, cnms_beta=cb, cnms_gamma=cg)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_thresh = float(thresh)
                            best_nms = int(nd)
                            best_top_n = int(tn)
                            best_anms_frac = float(af)
                            best_cnms = (ca, cb, cg)
    
    return best_thresh, best_nms, best_f1, best_top_n, best_anms_frac, best_atopn, best_cnms

def make_scheduler(optimizer, config, n_epochs):
    """Create LR scheduler from config."""
    sched_type = config.get('scheduler', 'cosine')
    warmup = config.get('warmup_epochs', 0)
    
    if sched_type == 'none':
        return None
    elif sched_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif sched_type == 'cosine_warmup':
        # Linear warmup then cosine
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(n_epochs - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sched_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)
    return None

def train(config: dict, train_dir: str, val_dir: str,
          val_adv_dir: str | None = None,
          device: str = 'cuda', verbose: bool = True) -> dict:
    """Full training loop.
    
    Args:
        config: full experiment config dict
        train_dir: path to training .pt files
        val_dir: path to validation .pt files
        val_adv_dir: optional path to adversarial validation .pt files
        device: 'cuda' or 'cpu'
        verbose: print progress
    
    Returns dict with:
        best_val_f1, best_val_rate_mae, final_metrics, training_history
    """
    # Load data — rebuild graphs if feature dims don't match config
    train_dir = _rebuild_graphs_if_needed(train_dir, "train", config, verbose)
    val_dir = _rebuild_graphs_if_needed(val_dir, "val", config, verbose)
    if val_adv_dir:
        val_adv_dir = _rebuild_graphs_if_needed(val_adv_dir, "val_adversarial", config, verbose)

    train_data = load_cached_graphs(train_dir)
    val_data = load_cached_graphs(val_dir)
    val_adv_data = load_cached_graphs(val_adv_dir) if val_adv_dir else []
    
    if verbose:
        print(f"Train: {len(train_data)} graphs, Val: {len(val_data)}, Val-Adv: {len(val_adv_data)}")
    
    # Build model
    in_dim, edge_dim = compute_feature_dims(config)
    model_config = {**config, 'in_dim': in_dim, 'edge_dim': edge_dim}
    model = build_model(model_config).to(device)
    
    if verbose:
        print(f"Model: {config.get('arch', 'gat')}, params: {count_parameters(model)}")
    
    # Data loaders
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    val_adv_loader = DataLoader(val_adv_data, batch_size=batch_size) if val_adv_data else None
    
    # Optimizer
    opt_name = config.get('optimizer', 'adamw')
    lr = config.get('lr', 1e-3)
    wd = config.get('weight_decay', 1e-4)
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'sgd_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    n_epochs = config.get('max_epochs', 100)
    patience = config.get('patience', 15)
    scheduler = make_scheduler(optimizer, config, n_epochs)
    tol = config.get('tol_samples_train', 75)
    
    best_val_f1 = 0.0
    best_metrics = {}
    best_model_state = None
    no_improve = 0
    history = []
    
    gpu_fell_back = False
    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss, device = train_one_epoch(model, train_loader, optimizer, config, device)
        if not gpu_fell_back and device == 'cpu' and config.get('_original_device', 'cuda') != 'cpu':
            gpu_fell_back = True
            print("  [WARN] Continuing training on CPU after GPU fallback")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device, tol_samples=tol)
        val_f1 = val_metrics['boundary_f1_600ms']
        
        # Adversarial eval
        if val_adv_loader:
            val_adv_metrics = evaluate(model, val_adv_loader, device, tol_samples=tol)
            # Weighted aggregate (55% val, 45% adv)
            agg_f1 = 0.55 * val_f1 + 0.45 * val_adv_metrics['boundary_f1_600ms']
        else:
            val_adv_metrics = {}
            agg_f1 = val_f1
        
        dt = time.time() - t0
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_f1': val_f1,
            'val_rate_mae': val_metrics['rate_mae_bpm'],
            'agg_f1': agg_f1,
            'lr': optimizer.param_groups[0]['lr'],
            'dt': dt,
        })
        
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} val_f1={val_f1:.4f} agg={agg_f1:.4f} lr={optimizer.param_groups[0]['lr']:.2e} ({dt:.1f}s)")
        
        # Track best
        if agg_f1 > best_val_f1:
            best_val_f1 = agg_f1
            best_metrics = {
                'boundary_f1_600ms': agg_f1,
                'val_f1': val_f1,
                'rate_mae_bpm': val_metrics['rate_mae_bpm'],
                'epoch': epoch,
            }
            if val_adv_metrics:
                best_metrics['val_adv_f1'] = val_adv_metrics['boundary_f1_600ms']
                best_metrics['val_adv_rate_mae'] = val_adv_metrics['rate_mae_bpm']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(agg_f1)
            else:
                scheduler.step()
        
        # Early stopping
        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (no improve for {patience} epochs)")
            break
    
    # Restore best model before post-processing optimization
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # --- SWA phase (optional) ---
    swa_start = config.get('swa_start_epoch', 0)
    swa_epochs = config.get('swa_epochs', 0)
    if swa_epochs > 0 and swa_start > 0:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_lr = config.get('swa_lr', 1e-4)
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        # Reset optimizer LR
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        if verbose:
            print(f"  SWA phase: {swa_epochs} epochs, swa_lr={swa_lr:.1e}")
        for swa_ep in range(swa_epochs):
            t0_swa = time.time()
            train_loss_swa, device = train_one_epoch(model, train_loader, optimizer, config, device)
            swa_model.update_parameters(model)
            swa_scheduler.step()
            dt_swa = time.time() - t0_swa
            if verbose and (swa_ep % 10 == 0 or swa_ep == swa_epochs - 1):
                print(f"  SWA {swa_ep:3d}/{swa_epochs}: loss={train_loss_swa:.4f} lr={optimizer.param_groups[0]['lr']:.2e} ({dt_swa:.1f}s)")
        # Update BN stats with SWA model
        try:
            update_bn(train_loader, swa_model, device=device)
        except Exception:
            pass  # No BN layers in GAT
        # Evaluate SWA model
        swa_metrics = evaluate(swa_model, val_loader, device, tol_samples=tol)
        swa_f1 = swa_metrics['boundary_f1_600ms']
        if verbose:
            print(f"  SWA val_f1={swa_f1:.4f} (pre-SWA best={best_metrics.get('val_f1', 0):.4f})")
        # Use SWA model if it's better or close (SWA models improve more with post-processing)
        if swa_f1 >= best_metrics.get('val_f1', 0) - 0.02:
            # Extract inner model weights from AveragedModel wrapper
            model.load_state_dict({k.replace('module.', ''): v for k, v in swa_model.state_dict().items() if k.startswith('module.')})
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  Using SWA model (swa_f1={swa_f1:.4f})")
        else:
            # Restore pre-SWA best
            model.load_state_dict(best_model_state)
            model.to(device)
            if verbose:
                print(f"  SWA model worse, keeping pre-SWA best")

    # Save checkpoint BEFORE post-processing (GPU may crash during threshold opt)
    checkpoint_dir = os.path.join(os.path.dirname(train_dir), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, 'best_model.pt')
    torch.save(best_model_state or model.state_dict(), ckpt_path)
    if verbose:
        print(f"  Saved checkpoint to {ckpt_path}")

    # Post-training: calibration + optimize threshold + NMS + top_n
    pp_device = 'cpu'
    model_cpu = model.cpu()
    wb = config.get('window_breaths', 6)
    use_adaptive = config.get('adaptive_nms', False)
    cal_method = config.get('calibration_method', 'none')
    mc_samples = config.get('mc_samples', 1)
    cal_params = None
    
    # Fit calibration on validation set
    if cal_method in ('temperature', 'platt'):
        raw_logits, raw_labels = _collect_raw_logits(
            model_cpu, val_loader, pp_device, mc_samples=mc_samples)
        if cal_method == 'temperature':
            T = learn_temperature(raw_logits, raw_labels)
            cal_params = {'temperature': T}
            if verbose:
                print(f"  Temperature scaling: T={T:.3f}")
        elif cal_method == 'platt':
            slope, intercept = fit_platt_scaling(raw_logits, raw_labels)
            cal_params = {'platt_slope': slope, 'platt_intercept': intercept}
            if verbose:
                print(f"  Platt scaling: slope={slope:.3f}, intercept={intercept:.3f}")
    
    use_cnms = config.get('composite_nms', False)
    opt_thresh, opt_nms, opt_f1, opt_top_n, opt_anms_frac, opt_atopn, opt_cnms = optimize_threshold(
        model_cpu, val_loader, pp_device, tol_samples=tol, window_breaths=wb,
        adaptive_nms=use_adaptive, mc_samples=mc_samples,
        calibration_method=cal_method, calibration_params=cal_params,
        composite_nms=use_cnms)
    opt_ca, opt_cb, opt_cg = opt_cnms
    if verbose:
        extra = ""
        if opt_anms_frac > 0:
            extra += f", anms_frac={opt_anms_frac:.2f}"
        if opt_atopn:
            extra += ", adaptive_top_n=True"
        if opt_cb > 0 or opt_cg > 0:
            extra += f", cnms=({opt_ca:.2f},{opt_cb:.3f},{opt_cg:.3f})"
        print(f"  Threshold opt: thresh={opt_thresh:.2f}, nms={opt_nms}, top_n={opt_top_n}{extra}, val_f1={opt_f1:.4f} (was {best_metrics.get('val_f1', 0):.4f})")

    # Re-evaluate with optimized threshold
    val_preds = _collect_predictions(model_cpu, val_loader, pp_device,
                                     mc_samples=mc_samples, calibration_method=cal_method,
                                     calibration_params=cal_params,
                                     include_features=use_cnms)
    final_val_f1 = _eval_threshold(val_preds, opt_thresh, opt_nms, opt_top_n, tol,
                                   adaptive_nms_frac=opt_anms_frac, adaptive_top_n=opt_atopn,
                                   cnms_alpha=opt_ca, cnms_beta=opt_cb, cnms_gamma=opt_cg)
    final_val_rate = evaluate(model_cpu, val_loader, pp_device, threshold=opt_thresh,
                              tol_samples=tol, nms_dist=opt_nms, top_n=opt_top_n)['rate_mae_bpm']
    
    if val_adv_loader:
        adv_preds = _collect_predictions(model_cpu, val_adv_loader, pp_device,
                                         mc_samples=mc_samples, calibration_method=cal_method,
                                         calibration_params=cal_params,
                                         include_features=use_cnms)
        final_adv_f1 = _eval_threshold(adv_preds, opt_thresh, opt_nms, opt_top_n, tol,
                                       adaptive_nms_frac=opt_anms_frac, adaptive_top_n=opt_atopn,
                                       cnms_alpha=opt_ca, cnms_beta=opt_cb, cnms_gamma=opt_cg)
        final_adv_rate = evaluate(model_cpu, val_adv_loader, pp_device, threshold=opt_thresh,
                                  tol_samples=tol, nms_dist=opt_nms, top_n=opt_top_n)['rate_mae_bpm']
        final_agg = 0.55 * final_val_f1 + 0.45 * final_adv_f1
    else:
        final_adv_f1 = None
        final_adv_rate = None
        final_agg = final_val_f1

    if final_agg > best_val_f1:
        best_val_f1 = final_agg
        best_metrics = {
            'boundary_f1_600ms': final_agg,
            'val_f1': final_val_f1,
            'rate_mae_bpm': final_val_rate,
            'epoch': best_metrics.get('epoch', len(history) - 1),
            'opt_threshold': opt_thresh,
            'opt_nms_dist': opt_nms,
            'opt_top_n': opt_top_n,
            'opt_anms_frac': opt_anms_frac,
            'opt_atopn': opt_atopn,
            'opt_cnms_alpha': opt_ca,
            'opt_cnms_beta': opt_cb,
            'opt_cnms_gamma': opt_cg,
            'calibration_method': cal_method,
            'mc_samples': mc_samples,
        }
        if cal_params:
            best_metrics['calibration_params'] = cal_params
        if final_adv_f1 is not None:
            best_metrics['val_adv_f1'] = final_adv_f1
            best_metrics['val_adv_rate_mae'] = final_adv_rate
        if verbose:
            print(f"  ★ Post-opt improved: agg_f1={final_agg:.4f}")

    return {
        'best_val_f1': best_val_f1,
        'best_metrics': best_metrics,
        'history': history,
        'n_epochs_run': len(history),
        'n_params': count_parameters(model),
        'opt_threshold': opt_thresh,
        'opt_nms_dist': opt_nms,
        'opt_top_n': opt_top_n,
        'opt_anms_frac': opt_anms_frac,
        'opt_atopn': opt_atopn,
        'opt_cnms_alpha': opt_ca,
        'opt_cnms_beta': opt_cb,
        'opt_cnms_gamma': opt_cg,
    }


def _train_single_model(config, train_loader, val_loader, val_adv_loader,
                         device, seed, verbose=True):
    """Train a single model with a given seed. Returns (best_model_state, raw_agg_f1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    in_dim, edge_dim = compute_feature_dims(config)
    model_config = {**config, 'in_dim': in_dim, 'edge_dim': edge_dim}
    model = build_model(model_config).to(device)
    
    lr = config.get('lr', 1e-3)
    wd = config.get('weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    n_epochs = config.get('max_epochs', 100)
    patience = config.get('patience', 15)
    scheduler = make_scheduler(optimizer, config, n_epochs)
    tol = config.get('tol_samples_train', 75)
    
    best_agg = 0.0
    best_state = None
    no_improve = 0
    
    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss, device = train_one_epoch(model, train_loader, optimizer, config, device)
        val_metrics = evaluate(model, val_loader, device, tol_samples=tol)
        val_f1 = val_metrics['boundary_f1_600ms']
        
        if val_adv_loader:
            adv_metrics = evaluate(model, val_adv_loader, device, tol_samples=tol)
            agg_f1 = 0.55 * val_f1 + 0.45 * adv_metrics['boundary_f1_600ms']
        else:
            agg_f1 = val_f1
        
        if agg_f1 > best_agg:
            best_agg = agg_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(agg_f1)
            else:
                scheduler.step()
        
        dt = time.time() - t0
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"    [seed={seed}] Epoch {epoch:3d}: agg={agg_f1:.4f} best={best_agg:.4f} ({dt:.1f}s)")
        
        if no_improve >= patience:
            if verbose:
                print(f"    [seed={seed}] Early stop at epoch {epoch}")
            break
    
    return best_state, best_agg


def train_ensemble(config: dict, train_dir: str, val_dir: str,
                   val_adv_dir: str | None = None,
                   device: str = 'cuda', verbose: bool = True) -> dict:
    """Train N models with different seeds, ensemble their scores for post-processing."""
    n_ensemble = config.get('n_ensemble', 5)
    
    train_dir = _rebuild_graphs_if_needed(train_dir, "train", config, verbose)
    val_dir = _rebuild_graphs_if_needed(val_dir, "val", config, verbose)
    if val_adv_dir:
        val_adv_dir = _rebuild_graphs_if_needed(val_adv_dir, "val_adversarial", config, verbose)
    
    train_data = load_cached_graphs(train_dir)
    val_data = load_cached_graphs(val_dir)
    val_adv_data = load_cached_graphs(val_adv_dir) if val_adv_dir else []
    
    if verbose:
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Val-Adv: {len(val_adv_data)}")
        in_dim, edge_dim = compute_feature_dims(config)
        model_config = {**config, 'in_dim': in_dim, 'edge_dim': edge_dim}
        print(f"Model: {config.get('arch', 'gat')}, params: {count_parameters(build_model(model_config))}")
    
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    val_adv_loader = DataLoader(val_adv_data, batch_size=batch_size) if val_adv_data else None
    
    checkpoint_dir = os.path.join(os.path.dirname(train_dir), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    all_states = []
    all_aggs = []
    for seed in range(n_ensemble):
        if verbose:
            print(f"  === Ensemble member {seed+1}/{n_ensemble} (seed={seed}) ===")
        state, agg = _train_single_model(
            config, train_loader, val_loader, val_adv_loader, device, seed, verbose)
        all_states.append(state)
        all_aggs.append(agg)
        torch.save(state, os.path.join(checkpoint_dir, f'ensemble_{seed}.pt'))
        if verbose:
            print(f"    Raw agg_f1={agg:.4f}")
    
    best_idx = int(np.argmax(all_aggs))
    torch.save(all_states[best_idx], os.path.join(checkpoint_dir, 'best_model.pt'))
    if verbose:
        print(f"  Ensemble raw: mean={np.mean(all_aggs):.4f}, best={max(all_aggs):.4f} (seed={best_idx})")
    
    # Collect predictions from ALL models, average scores
    in_dim, edge_dim = compute_feature_dims(config)
    model_config = {**config, 'in_dim': in_dim, 'edge_dim': edge_dim}
    pp_device = 'cpu'
    tol = config.get('tol_samples_train', 75)
    wb = config.get('window_breaths', 6)
    
    def _ensemble_preds(loader, states):
        all_preds = []
        for state in states:
            m = build_model(model_config)
            m.load_state_dict(state)
            m.to(pp_device).eval()
            all_preds.append(_collect_predictions(m, loader, pp_device))
        n_g = len(all_preds[0])
        merged = []
        for g in range(n_g):
            avg = np.mean([p[g][0] for p in all_preds], axis=0)
            merged.append((avg, all_preds[0][g][1], all_preds[0][g][2]))
        return merged
    
    ens_val = _ensemble_preds(val_loader, all_states)
    
    # Grid search on ensemble predictions
    best_f1 = 0.0
    best_thresh = 0.5
    best_nms = 0
    best_top_n = 0
    
    for thresh in np.arange(0.05, 0.90, 0.05):
        for nd in [0, 50, 75, 100, 120, 150, 200]:
            for tn in [0, wb]:
                f1 = _eval_threshold(ens_val, thresh, nd, tn, tol)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(thresh)
                    best_nms = int(nd)
                    best_top_n = int(tn)
    
    for thresh in np.arange(max(0.02, best_thresh - 0.10),
                            min(0.95, best_thresh + 0.10), 0.01):
        for nd in range(max(0, best_nms - 30), best_nms + 35, 5):
            for tn in [0, wb]:
                f1 = _eval_threshold(ens_val, thresh, nd, tn, tol)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(thresh)
                    best_nms = int(nd)
                    best_top_n = int(tn)
    
    if verbose:
        print(f"  Ensemble opt: thresh={best_thresh:.2f}, nms={best_nms}, top_n={best_top_n}, F1={best_f1:.4f}")
    
    # Best single model for comparison
    best_single_preds = _collect_predictions(
        build_model(model_config).cpu().eval() if False else
        (lambda: (m := build_model(model_config), m.load_state_dict(all_states[best_idx]),
                  m.to(pp_device).eval(), m)[-1])(),
        val_loader, pp_device)
    best_single_f1 = 0.0
    for thresh in np.arange(0.05, 0.90, 0.05):
        for nd in [0, 50, 75, 100, 120, 150, 200]:
            for tn in [0, wb]:
                f1 = _eval_threshold(best_single_preds, thresh, nd, tn, tol)
                best_single_f1 = max(best_single_f1, f1)
    if verbose:
        print(f"  Best single F1={best_single_f1:.4f} vs ensemble F1={best_f1:.4f}")
    
    # Evaluate on val_adv
    final_val_f1 = best_f1
    final_adv_f1 = None
    if val_adv_loader:
        ens_adv = _ensemble_preds(val_adv_loader, all_states)
        final_adv_f1 = _eval_threshold(ens_adv, best_thresh, best_nms, best_top_n, tol)
        final_agg = 0.55 * final_val_f1 + 0.45 * final_adv_f1
    else:
        final_agg = final_val_f1
    
    if verbose:
        print(f"  Final: val={final_val_f1:.4f}, adv={final_adv_f1}, agg={final_agg:.4f}")
    
    best_metrics = {
        'boundary_f1_600ms': final_agg,
        'val_f1': final_val_f1,
        'rate_mae_bpm': 0.0,
        'opt_threshold': best_thresh,
        'opt_nms_dist': best_nms,
        'opt_top_n': best_top_n,
        'n_ensemble': n_ensemble,
        'ensemble_raw_aggs': all_aggs,
        'best_single_f1': best_single_f1,
    }
    if final_adv_f1 is not None:
        best_metrics['val_adv_f1'] = final_adv_f1
    
    return {
        'best_val_f1': final_agg,
        'best_metrics': best_metrics,
        'history': [],
        'n_epochs_run': n_ensemble,
        'n_params': count_parameters(build_model(model_config)),
        'opt_threshold': best_thresh,
        'opt_nms_dist': best_nms,
        'opt_top_n': best_top_n,
        'opt_anms_frac': 0.0,
        'opt_atopn': False,
    }
