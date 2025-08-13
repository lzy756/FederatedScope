#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize the solving process and effects from a solution log.

Inputs: a text log like data/solution/solution.txt containing lines such as:
 - Feature matrix P shape: (10, 10)
 - PC #i (n clients): [ ... 10 numbers ...]
 - Target distribution: [ ... 10 numbers ...]
 - Matrix A shape: (10, 10)
 - Vector M shape: (10,)
 - Upper bounds B: [ ... 10 ints ...]
 - Total clusters K: 10
 - Total clients to select S: 50
 - Binary decomposition: G = [5, 5, ...]
 - Gurobi solved successfully with objective value: ...
 - Feature match score: ...
 - Max absolute difference: ...
 - L2 norm of differences: ...
 - KL divergence: ...
 - Predicted features: [ ... 10 numbers ...]
 - Target features: [ ... 10 numbers ...]
 - Absolute difference: [ ... 10 numbers ...]
 - Relative difference: [ ... 10 numbers ...]
 - Template solved successfully: h = [ ... 10 numbers ...]
 - Selected clients ... for round r:
   Cluster #c: m clients - [id, id, ...]

Outputs:
 - A_heatmap.png: Heatmap of A (features x clusters) rebuilt from PC vectors.
 - B_h_bar.png: Side-by-side bars of B (cluster sizes/upper bounds) vs h (selected per cluster).
 - target_vs_pred_features.png: Grouped bars per feature comparing target vs predicted features.
 - diffs_abs_rel.png: Absolute and relative differences per feature.
 - pc_feature_heatmap.png: Heatmap of per-cluster feature vectors (PCs).
 - selection_round0.png: Selected counts per cluster in round 0 (if available).
 - summary.txt: Parsed key metrics.

Usage:
  python -m federatedscope.contrib.tools.visualize_template_solving \
      --log data/solution/solution.txt \
      --outdir data/solution/figs \
      --show 0
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _parse_array(text: str) -> np.ndarray:
    # extract numbers within brackets and split by comma/space
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return np.array([float(x) for x in nums], dtype=float)


def parse_solution_log(path: str) -> Dict:
    data: Dict = {
        "pc_vectors": [],  # list of np.ndarray, length F per PC
        "target_distribution": None,  # np.ndarray length F
        "A": None,
        "B": None,  # np.ndarray length K
        "K": None,
        "S": None,
        "G": None,  # np.ndarray length K (binary decomposition group sizes)
        "objective": None,
        "feat_match": None,  # dict with percent, matched, total, tol
        "max_abs_diff": None,
        "l2_norm": None,
        "kl_div": None,
        "pred_features": None,
        "tgt_features": None,
        "abs_diff": None,
        "rel_diff": None,
        "h": None,
        "selection_rounds": {},  # round -> {cluster_idx: [ids]}
    }

    pc_order: List[int] = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Target distribution
            if 'Target distribution' in line:
                arr = _parse_array(line)
                data["target_distribution"] = arr
            # PC lines
            m = re.search(r"PC\s*#(\d+)\s*\(.*?\):\s*\[(.*)\]", line)
            if m:
                idx = int(m.group(1)) - 1  # convert to 0-based
                arr = _parse_array(m.group(2))
                # ensure list long enough
                while len(data["pc_vectors"]) <= idx:
                    data["pc_vectors"].append(None)
                data["pc_vectors"][idx] = arr
                pc_order.append(idx)
                continue
            # Matrix shapes
            if 'Matrix A shape' in line:
                ms = re.search(r"Matrix A shape: \((\d+),\s*(\d+)\)", line)
                if ms:
                    data["A_shape"] = (int(ms.group(1)), int(ms.group(2)))
            if 'Vector M shape' in line:
                ms = re.search(r"Vector M shape: \((\d+)\,?\)", line)
                if ms:
                    data["M_shape"] = int(ms.group(1))
            # B, K, S
            if 'Upper bounds B' in line:
                data["B"] = _parse_array(line)
            if 'Total clusters K' in line:
                ms = re.search(r"Total clusters K:\s*(\d+)", line)
                if ms:
                    data["K"] = int(ms.group(1))
            if 'Total clients to select S' in line:
                ms = re.search(r"Total clients to select S:\s*(\d+)", line)
                if ms:
                    data["S"] = int(ms.group(1))
            if 'Binary decomposition: G' in line:
                data["G"] = _parse_array(line)
            if 'Gurobi solved successfully with objective value' in line:
                ms = re.search(r"objective value:\s*([\d\.eE+\-]+)", line)
                if ms:
                    data["objective"] = float(ms.group(1))
            if 'Feature match score' in line:
                ms = re.search(r"Feature match score:\s*([\d\.]+)%\s*\((\d+)/(\d+) features matched within ([\d\.]+)% error\)", line)
                if ms:
                    data["feat_match"] = {
                        'percent': float(ms.group(1)),
                        'matched': int(ms.group(2)),
                        'total': int(ms.group(3)),
                        'tol_percent': float(ms.group(4)),
                    }
            if line.startswith('Max absolute difference:'):
                data["max_abs_diff"] = float(_parse_array(line)[0])
            if line.startswith('L2 norm of differences:'):
                data["l2_norm"] = float(_parse_array(line)[0])
            if line.startswith('KL divergence:'):
                data["kl_div"] = float(_parse_array(line)[0])
            if line.startswith('Predicted features:'):
                data["pred_features"] = _parse_array(line)
            if line.startswith('Target features:'):
                data["tgt_features"] = _parse_array(line)
            if line.startswith('Absolute difference:'):
                data["abs_diff"] = _parse_array(line)
            if line.startswith('Relative difference:'):
                data["rel_diff"] = _parse_array(line)
            if 'Template solved successfully: h =' in line:
                data["h"] = _parse_array(line)
            # selection per round
            round_hdr = re.search(r"Selected clients from (\d+) clusters for round (\d+):", line)
            if round_hdr:
                # following lines contain cluster details; we capture in subsequent iterations
                current_round = int(round_hdr.group(2))
                data["selection_rounds"][current_round] = {}
                continue
            m2 = re.search(r"Cluster\s*#(\d+):\s*(\d+) clients\s*-\s*\[(.*)\]", line)
            if m2:
                cidx = int(m2.group(1))
                ids = _parse_array(m2.group(3)).astype(int).tolist()
                # try to assign to the last recorded round (max key)
                if data["selection_rounds"]:
                    rr = max(data["selection_rounds"].keys())
                    data["selection_rounds"][rr][cidx] = ids
                continue

    # Build A from pc_vectors if possible
    pcs: List[Optional[np.ndarray]] = data["pc_vectors"]
    if pcs and all(p is not None for p in pcs):
        # Each PC vector is length F; assemble as columns to shape (F, K)
        A = np.stack(pcs, axis=1)  # (F, K)
        data["A"] = A
        data["F"] = A.shape[0]
        data["K"] = A.shape[1] if data.get("K") is None else data["K"]
    return data


def plot_A_heatmap(A: np.ndarray, outpath: str, show: bool = False):
    if A is None or A.size == 0:
        return
    plt.figure(figsize=(8, 6))
    im = plt.imshow(A, aspect='auto', cmap='magma')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Feature contribution')
    plt.xlabel('Cluster index (PC)')
    plt.ylabel('Feature index')
    plt.title('A matrix heatmap (features x clusters)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_pc_feature_heatmap(A: np.ndarray, outpath: str, show: bool = False):
    if A is None or A.size == 0:
        return
    plt.figure(figsize=(8, 6))
    im = plt.imshow(A.T, aspect='auto', cmap='viridis')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Feature contribution')
    plt.xlabel('Feature index')
    plt.ylabel('Cluster index (PC)')
    plt.title('Per-cluster feature vectors (PCs)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_B_h(B: Optional[np.ndarray], h: Optional[np.ndarray], outpath: str, show: bool = False):
    if B is None and h is None:
        return
    if B is None:
        B = np.zeros_like(h)
    if h is None:
        h = np.zeros_like(B)
    K = max(len(B), len(h))
    x = np.arange(K)
    plt.figure(figsize=(max(8, K * 0.6), 4))
    width = 0.38
    plt.bar(x - width/2, B, width=width, label='B (upper bound / cluster size)')
    plt.bar(x + width/2, h, width=width, label='h (selected per cluster)')
    for i in range(K):
        plt.text(i - width/2, B[i] + 0.5, f"{int(B[i])}", ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, h[i] + 0.5, f"{int(h[i])}", ha='center', va='bottom', fontsize=8)
    plt.xlabel('Cluster index (0-based)')
    plt.ylabel('#Clients')
    plt.title('Cluster capacity (B) vs selected counts (h)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_target_vs_pred(tgt: Optional[np.ndarray], pred: Optional[np.ndarray], outpath: str, show: bool = False):
    if tgt is None or pred is None:
        return
    F = len(tgt)
    x = np.arange(F)
    plt.figure(figsize=(max(8, F * 0.6), 4))
    width = 0.38
    plt.bar(x - width/2, tgt, width=width, label='Target features (M)')
    plt.bar(x + width/2, pred, width=width, label='Predicted features (A@h)')
    plt.xlabel('Feature index')
    plt.ylabel('Counts / contribution')
    plt.title('Target vs Predicted features')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_diffs(abs_diff: Optional[np.ndarray], rel_diff: Optional[np.ndarray], outpath: str, show: bool = False):
    if abs_diff is None and rel_diff is None:
        return
    if abs_diff is None:
        abs_diff = np.zeros_like(rel_diff)
    if rel_diff is None:
        rel_diff = np.zeros_like(abs_diff)
    F = len(abs_diff)
    x = np.arange(F)
    fig, ax1 = plt.subplots(figsize=(max(8, F * 0.6), 4))
    ax1.bar(x, abs_diff, color='tab:blue', label='Absolute diff')
    ax1.set_xlabel('Feature index')
    ax1.set_ylabel('Absolute diff', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(x, rel_diff, color='tab:red', marker='o', label='Relative diff')
    ax2.set_ylabel('Relative diff', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.suptitle('Feature-wise residuals')
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _safe_normalize(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    v = np.array(v, dtype=float)
    s = v.sum()
    if s <= 0:
        return None
    return v / s


def plot_distribution_match(tgt: Optional[np.ndarray], pred: Optional[np.ndarray], kl_div: Optional[float], outpath: str, show: bool = False):
    """Plot normalized target vs predicted distributions with divergence metrics.
    Args:
        tgt: target feature counts (length F)
        pred: predicted feature counts (length F)
        kl_div: KL divergence value from log (optional)
    """
    pt = _safe_normalize(tgt)
    pp = _safe_normalize(pred)
    if pt is None or pp is None:
        return
    F = len(pt)
    x = np.arange(F)
    # distances
    l1 = float(np.sum(np.abs(pt - pp)))
    tv = 0.5 * l1
    l2 = float(np.linalg.norm(pt - pp))
    # compute KL if not provided (add small eps for stability)
    if kl_div is None:
        eps = 1e-12
        q = np.clip(pp, eps, 1.0)
        p = np.clip(pt, eps, 1.0)
        kl_div = float(np.sum(p * np.log(p / q)))

    plt.figure(figsize=(max(8, F * 0.6), 4.5))
    width = 0.4
    plt.bar(x - width/2, pt, width=width, label='Target dist', color='tab:blue')
    plt.bar(x + width/2, pp, width=width, label='Selected dist', color='tab:orange')
    # annotate metrics in the title
    plt.xlabel('Feature index')
    plt.ylabel('Probability')
    plt.title(f'Distribution match: KL={kl_div:.4f}, TV={tv:.4f}, L2={l2:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_selection_round(selection_for_round: Dict[int, List[int]], round_idx: int, outpath: str, show: bool = False):
    if not selection_for_round:
        return
    K = max(selection_for_round.keys()) + 1 if selection_for_round else 0
    counts = np.zeros(K, dtype=int)
    for k, ids in selection_for_round.items():
        counts[k] = len(ids)
    x = np.arange(K)
    plt.figure(figsize=(max(8, K * 0.6), 4))
    plt.bar(x, counts, color='tab:green')
    for i in range(K):
        plt.text(i, counts[i] + 0.3, str(counts[i]), ha='center', va='bottom', fontsize=8)
    plt.xlabel('Cluster index (as logged)')
    plt.ylabel(f'#Selected clients (round {round_idx})')
    plt.title(f'Selected clients per cluster (round {round_idx})')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def write_summary_txt(parsed: Dict, outpath: str):
    # compute fallback metrics if missing
    tgt = parsed.get('tgt_features')
    pred = parsed.get('pred_features')
    max_abs = parsed.get('max_abs_diff')
    l2n = parsed.get('l2_norm')
    kld = parsed.get('kl_div')
    try:
        if tgt is not None and pred is not None:
            tgt = np.array(tgt, dtype=float)
            pred = np.array(pred, dtype=float)
            diff = np.abs(pred - tgt)
            if max_abs is None:
                max_abs = float(np.max(diff))
                parsed['max_abs_diff'] = max_abs
            if l2n is None:
                l2n = float(np.linalg.norm(pred - tgt))
                parsed['l2_norm'] = l2n
            if kld is None:
                # KL on normalized distributions
                pt = tgt / (tgt.sum() if tgt.sum() != 0 else 1.0)
                pp = pred / (pred.sum() if pred.sum() != 0 else 1.0)
                eps = 1e-12
                p = np.clip(pt, eps, 1.0)
                q = np.clip(pp, eps, 1.0)
                kld = float(np.sum(p * np.log(p / q)))
                parsed['kl_div'] = kld
    except Exception:
        pass

    lines = []
    def add(k, v):
        lines.append(f"{k}: {v}")
    add('Objective', parsed.get('objective'))
    fm = parsed.get('feat_match')
    if fm:
        add('Feature match', f"{fm['percent']}% ({fm['matched']}/{fm['total']} within {fm['tol_percent']}% error)")
    add('Max absolute diff', parsed.get('max_abs_diff'))
    add('L2 norm diff', parsed.get('l2_norm'))
    add('KL divergence', parsed.get('kl_div'))
    add('K (clusters)', parsed.get('K'))
    add('S (total to select)', parsed.get('S'))
    # Save
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main(log_path: str, outdir: Optional[str] = None, show: bool = False):
    parsed = parse_solution_log(log_path)
    if outdir is None:
        outdir = os.path.join(os.path.dirname(log_path) or '.', 'figs')
    _ensure_outdir(outdir)

    A = parsed.get('A')
    B = parsed.get('B')
    h = parsed.get('h')
    tgt = parsed.get('tgt_features')
    pred = parsed.get('pred_features')
    abs_diff = parsed.get('abs_diff')
    rel_diff = parsed.get('rel_diff')
    selection_rounds = parsed.get('selection_rounds', {})

    # Plots
    plot_A_heatmap(A, os.path.join(outdir, 'A_heatmap.png'), show)
    plot_pc_feature_heatmap(A, os.path.join(outdir, 'pc_feature_heatmap.png'), show)
    plot_B_h(B, h, os.path.join(outdir, 'B_h_bar.png'), show)
    plot_target_vs_pred(tgt, pred, os.path.join(outdir, 'target_vs_pred_features.png'), show)
    plot_diffs(abs_diff, rel_diff, os.path.join(outdir, 'diffs_abs_rel.png'), show)
    if selection_rounds:
        first_round = min(selection_rounds.keys())
        sel_map = selection_rounds.get(first_round)
        plot_selection_round(sel_map, first_round, os.path.join(outdir, f'selection_round{first_round}.png'), show)
    # Distribution match plot (probability)
    plot_distribution_match(tgt, pred, parsed.get('kl_div'), os.path.join(outdir, 'target_vs_selected_distribution.png'), show)

    write_summary_txt(parsed, os.path.join(outdir, 'summary.txt'))
    print(f"Figures saved to: {outdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize solving process and effects from a solution log.')
    parser.add_argument('--log', type=str, default='data/solution/solution.txt', help='Path to solution log file')
    parser.add_argument('--outdir', type=str, default=None, help='Directory to save figures')
    parser.add_argument('--show', type=int, default=0, help='Show figures interactively (1=yes)')
    args = parser.parse_args()
    main(args.log, args.outdir, bool(args.show))
