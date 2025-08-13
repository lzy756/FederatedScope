#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize client data distribution differences and peer communities.

Inputs: a JSON file saved by FedGS_Splitter.save_cluster_results, e.g., data/peer_communities.json

It will produce:
 - label_distribution_stacked_bar.png: stacked bar chart of per-client label distributions
 - distribution_pca_scatter.png: PCA scatter plot of clients colored by community
 - distance_matrix_heatmap.png: heatmap of pairwise distances (Euclidean) between clients
 - group_sizes_bar.png: bar chart of sizes of each community

Usage:
  python -m federatedscope.contrib.tools.visualize_peer_communities \
      --json data/peer_communities.json \
      --outdir data/peer_communities_figs \
      --show 0
"""

import argparse
import json
import os
import random
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_peer_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    train_dist = data.get("train_distributions", [])
    test_dist = data.get("test_distributions", [])
    communities = data.get("peer_communities", [])

    # Convert to numpy arrays
    train_dist = np.array(train_dist, dtype=float) if len(train_dist) > 0 else np.zeros((0, 0))
    test_dist = np.array(test_dist, dtype=float) if len(test_dist) > 0 else np.zeros((0, 0))
    # Communities are lists of 0-based client indices; keep as list of lists
    return meta, train_dist, test_dist, communities


def normalize_distributions(dists: np.ndarray, mode: str = "l1") -> np.ndarray:
    if dists.size == 0:
        return dists
    d = dists.copy().astype(float)
    if mode == "l2":
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return d / norms
    # default l1 to make proportions
    sums = d.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return d / sums


def plot_label_distribution_stacked(train_dist: np.ndarray, communities: List[List[int]], outpath: str, show: bool = False):
    if train_dist.size == 0:
        return
    proportions = normalize_distributions(train_dist, mode="l1")
    n_clients, n_classes = proportions.shape

    # Order clients by community for better visualization
    ordered_clients = []
    comm_ids = []
    for cid, group in enumerate(communities):
        for cli in group:
            ordered_clients.append(cli)
            comm_ids.append(cid)
    remaining = [i for i in range(n_clients) if i not in ordered_clients]
    ordered_clients.extend(remaining)
    comm_ids.extend([-1] * len(remaining))

    ordered_mat = proportions[ordered_clients]

    plt.figure(figsize=(max(10, n_clients * 0.4), 5 + 0.1 * n_classes))
    bottom = np.zeros(len(ordered_clients))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes)) if n_classes <= 20 else plt.cm.nipy_spectral(np.linspace(0, 1, n_classes))
    for c in range(n_classes):
        plt.bar(range(len(ordered_clients)), ordered_mat[:, c], bottom=bottom, color=colors[c], edgecolor='none', label=f"C{c}")
        bottom += ordered_mat[:, c]

    # vertical lines to separate communities
    cursor = 0
    for group in communities:
        if not group:
            continue
        cursor += len(group)
        plt.axvline(cursor - 0.5, color='k', linestyle='--', alpha=0.3)

    plt.xlabel('Clients (ordered by community, 1-based IDs in title)')
    plt.ylabel('Label proportion')
    plt.title('Per-client label distributions (stacked)')
    # reduce legend clutter for many classes
    if n_classes <= 15:
        plt.legend(ncol=min(5, n_classes), fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_pca_scatter(train_dist: np.ndarray, communities: List[List[int]], outpath: str, show: bool = False):
    if train_dist.size == 0 or train_dist.shape[0] < 2:
        return
    props = normalize_distributions(train_dist, mode="l1")
    pca = PCA(n_components=2, random_state=0)
    emb = pca.fit_transform(props)

    plt.figure(figsize=(7, 6))
    # Assign colors per community
    n_comm = max(1, len(communities))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_comm, 3)))
    used = np.zeros(len(props), dtype=bool)
    for idx, group in enumerate(communities):
        if len(group) == 0:
            continue
        pts = emb[group]
        plt.scatter(pts[:, 0], pts[:, 1], c=[colors[idx % len(colors)]], label=f"PC{idx} ({len(group)})", s=40, alpha=0.85)
        used[group] = True
    # others
    others = np.where(~used)[0]
    if len(others) > 0:
        pts = emb[others]
        plt.scatter(pts[:, 0], pts[:, 1], c='gray', label=f"Unassigned ({len(others)})", s=40, alpha=0.6)
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title('PCA of label distributions (clients)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_distance_heatmap(train_dist: np.ndarray, communities: List[List[int]], outpath: str, show: bool = False):
    if train_dist.size == 0:
        return
    # Use L2 on L1-normalized proportions to reflect shape differences
    props = normalize_distributions(train_dist, mode="l1")
    dist = pairwise_distances(props, metric='euclidean')

    # order by community
    order = []
    for group in communities:
        order.extend(group)
    order.extend([i for i in range(props.shape[0]) if i not in order])
    dist_ordered = dist[np.ix_(order, order)]

    plt.figure(figsize=(7, 6))
    im = plt.imshow(dist_ordered, cmap='viridis')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Euclidean distance')
    # grid lines separate communities
    cursor = 0
    for group in communities:
        if not group:
            continue
        cursor += len(group)
        plt.axhline(cursor - 0.5, color='w', linestyle='--', linewidth=0.7, alpha=0.6)
        plt.axvline(cursor - 0.5, color='w', linestyle='--', linewidth=0.7, alpha=0.6)
    plt.title('Client distance matrix (ordered by community)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_group_sizes(communities: List[List[int]], outpath: str, show: bool = False, sampled_clients: Optional[Sequence[int]] = None):
    sizes = [len(g) for g in communities]
    if len(sizes) == 0:
        return
    plt.figure(figsize=(max(6, 1 + 0.6 * len(sizes)), 4))
    bars = plt.bar(range(len(sizes)), sizes, color=plt.cm.tab10(np.linspace(0, 1, max(len(sizes), 3))))
    plt.xlabel('Peer community index')
    plt.ylabel('#Clients')
    plt.title('Sizes of peer communities')
    for i, s in enumerate(sizes):
        label = str(s)
        if sampled_clients is not None and i < len(sampled_clients) and sampled_clients[i] is not None:
            label = f"{s} (cid {sampled_clients[i] + 1})"  # show 1-based id
        plt.text(i, s + max(0.1, 0.02 * max(sizes)), label, ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def sample_one_per_community(communities: List[List[int]], seed: Optional[int] = None) -> List[Optional[int]]:
    rng = random.Random(seed)
    sampled = []
    for g in communities:
        if len(g) == 0:
            sampled.append(None)
        else:
            sampled.append(rng.choice(g))
    return sampled


def plot_sampled_clients_label_distribution(train_dist: np.ndarray,
                                            communities: List[List[int]],
                                            sampled_clients: Sequence[Optional[int]],
                                            outpath: str,
                                            show: bool = False):
    if train_dist.size == 0:
        return
    indices = [cid for cid in sampled_clients if cid is not None]
    if len(indices) == 0:
        return
    props = normalize_distributions(train_dist, mode="l1")
    sub = props[indices]
    n_sel, n_classes = sub.shape
    plt.figure(figsize=(max(8, n_sel * 0.8), 4 + 0.1 * n_classes))
    bottom = np.zeros(n_sel)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes)) if n_classes <= 20 else plt.cm.nipy_spectral(np.linspace(0, 1, n_classes))
    for c in range(n_classes):
        plt.bar(range(n_sel), sub[:, c], bottom=bottom, color=colors[c], edgecolor='none', label=f"C{c}")
        bottom += sub[:, c]
    xticklabels = []
    # map each sampled index to its community id
    cid_to_comm = {}
    for ci, g in enumerate(communities):
        for elem in g:
            cid_to_comm[elem] = ci
    for cid in indices:
        xticklabels.append(f"PC{cid_to_comm.get(cid, -1)}-cid{cid + 1}")
    plt.xticks(range(n_sel), xticklabels, rotation=45, ha='right')
    plt.ylabel('Label proportion')
    plt.title('Sampled one client per community: label distributions')
    if n_classes <= 15:
        plt.legend(ncol=min(5, n_classes), fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


def main(json_path: str, outdir: Optional[str] = None, show: bool = False, sample_one: bool = False, seed: Optional[int] = None):
    meta, train_dist, test_dist, communities = load_peer_json(json_path)
    if outdir is None:
        outdir = os.path.join(os.path.dirname(json_path) or ".", "peer_communities_figs")
    _ensure_outdir(outdir)

    # Basic info
    n_clients = train_dist.shape[0]
    n_classes = train_dist.shape[1] if train_dist.size > 0 else 0
    print(f"Loaded: {json_path}")
    print(f"Clients: {n_clients}, Classes: {n_classes}, Communities: {len(communities)}")

    # Plots
    plot_label_distribution_stacked(train_dist, communities, os.path.join(outdir, "label_distribution_stacked_bar.png"), show)
    plot_pca_scatter(train_dist, communities, os.path.join(outdir, "distribution_pca_scatter.png"), show)
    plot_distance_heatmap(train_dist, communities, os.path.join(outdir, "distance_matrix_heatmap.png"), show)

    sampled = None
    if sample_one:
        sampled = sample_one_per_community(communities, seed=seed)
        # annotate bar chart with sampled clients
        plot_group_sizes(communities, os.path.join(outdir, "group_sizes_bar.png"), show, sampled_clients=sampled)
        # also plot the sampled clients' label distributions
        plot_sampled_clients_label_distribution(
            train_dist, communities, sampled,
            os.path.join(outdir, "sampled_clients_label_distribution.png"), show
        )
    else:
        plot_group_sizes(communities, os.path.join(outdir, "group_sizes_bar.png"), show)

    print(f"Figures saved to: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize client distributions and peer communities.")
    parser.add_argument("--json", type=str, default="data/peer_communities.json", help="Path to peer_communities.json")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save figures")
    parser.add_argument("--show", type=int, default=0, help="Whether to show figures interactively (1=yes, 0=no)")
    parser.add_argument("--sample_one_per_comm", type=int, default=0, help="Sample one client per community and annotate/plot")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()
    main(args.json, args.outdir, bool(args.show), bool(args.sample_one_per_comm), args.seed)
