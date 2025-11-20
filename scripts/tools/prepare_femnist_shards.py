#!/usr/bin/env python3
"""Utility to split FEMNIST tasks into per-client shards."""

import argparse
import json
import os
import os.path as osp
import random
import shutil
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create FEMNIST shards for distributed training')
    parser.add_argument('--source',
                        default='data/femnist/processed',
                        help='Directory containing task_* folders')
    parser.add_argument('--output',
                        default='data/femnist/shards',
                        help='Destination directory for shards')
    parser.add_argument('--clients',
                        type=int,
                        required=True,
                        help='Number of clients to generate')
    parser.add_argument('--strategy',
                        choices=['round_robin', 'random', 'manual'],
                        default='round_robin',
                        help='Task assignment strategy')
    parser.add_argument('--mapping',
                        default='',
                        help='Path to JSON mapping for manual strategy')
    parser.add_argument('--manifest',
                        default='',
                        help='Optional output path for manifest.json')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Overwrite existing shards if present')
    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='Random seed for random strategy')
    return parser.parse_args()


def collect_tasks(source_dir: str) -> List[Tuple[int, str]]:
    if not osp.isdir(source_dir):
        raise FileNotFoundError(f'Source directory {source_dir} not found')
    tasks = []
    for entry in os.listdir(source_dir):
        if not entry.startswith('task_'):
            continue
        try:
            idx = int(entry.split('_')[1])
        except (IndexError, ValueError):
            continue
        tasks.append((idx, osp.join(source_dir, entry)))
    if not tasks:
        raise RuntimeError(f'No task_* folders found under {source_dir}')
    tasks.sort(key=lambda x: x[0])
    return tasks


def assign_tasks(tasks: Sequence[Tuple[int, str]], num_clients: int,
                 strategy: str, mapping_file: str,
                 seed: int) -> Dict[int, List[Tuple[int, str]]]:
    assignment = {cid: [] for cid in range(1, num_clients + 1)}
    if strategy == 'manual':
        if not mapping_file:
            raise ValueError('Manual strategy requires --mapping JSON file')
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        task_lookup = {idx: path for idx, path in tasks}
        for key, task_ids in mapping.items():
            cid = int(key)
            if cid not in assignment:
                raise ValueError(f'Client {cid} not in 1..{num_clients}')
            for task_id in task_ids:
                task_id = int(task_id)
                if task_id not in task_lookup:
                    raise ValueError(f'Task {task_id} missing from source')
                assignment[cid].append((task_id, task_lookup[task_id]))
    else:
        ordered = list(tasks)
        if strategy == 'random':
            rng = random.Random(seed)
            rng.shuffle(ordered)
        for idx, task in enumerate(ordered):
            cid = (idx % num_clients) + 1
            assignment[cid].append(task)
    return assignment


def load_split(task_path: str, split: str):
    file_path = osp.join(task_path, f'{split}.pt')
    if not osp.exists(file_path):
        return None
    return torch.load(file_path)


def aggregate_split(task_paths: Sequence[str], split: str):
    data_parts, target_parts = [], []
    for task_path in task_paths:
        split_pair = load_split(task_path, split)
        if split_pair is None:
            continue
        data, targets = split_pair
        data_parts.append(data)
        target_parts.append(targets)
    if not data_parts:
        return None
    data_tensor = torch.cat(data_parts, dim=0)
    target_tensor = torch.cat(target_parts, dim=0)
    return data_tensor, target_tensor


def build_label_hist(target_tensor: torch.Tensor) -> Dict[str, int]:
    if target_tensor is None:
        return {}
    counter = Counter()
    for value in target_tensor.tolist():
        counter[str(int(value))] += 1
    return dict(counter)


def save_split(output_dir: str, name: str, payload):
    if payload is None:
        return
    torch.save(payload, osp.join(output_dir, f'{name}.pt'))


def materialize_shard(cid: int, tasks: Sequence[Tuple[int, str]],
                      output_root: str, overwrite: bool):
    shard_dir = osp.join(output_root, f'client_{cid}')
    if osp.isdir(shard_dir):
        if overwrite:
            shutil.rmtree(shard_dir)
        else:
            raise FileExistsError(
                f'{shard_dir} exists. Use --overwrite to rebuild it.')
    os.makedirs(shard_dir, exist_ok=True)
    task_paths = [path for _, path in tasks]
    train_pair = aggregate_split(task_paths, 'train')
    if train_pair is None:
        raise RuntimeError(f'Client {cid} has no train samples')
    val_pair = aggregate_split(task_paths, 'val')
    test_pair = aggregate_split(task_paths, 'test')
    save_split(shard_dir, 'train', train_pair)
    save_split(shard_dir, 'val', val_pair)
    save_split(shard_dir, 'test', test_pair)
    hist_tensor = train_pair[1]
    metadata = {
        'client_id': cid,
        'tasks': [idx for idx, _ in tasks],
        'num_samples': {
            'train': int(train_pair[1].shape[0]),
            'val': int(val_pair[1].shape[0]) if val_pair else 0,
            'test': int(test_pair[1].shape[0]) if test_pair else 0,
        },
        'label_hist': build_label_hist(hist_tensor)
    }
    with open(osp.join(shard_dir, 'meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    return shard_dir, metadata


def write_manifest(manifest_path: str, dataset: str, source: str,
                   strategy: str, requested_clients: int,
                   assignment_meta: Dict[int, Dict[str, object]]):
    manifest = {
        'dataset': dataset,
        'source': osp.abspath(source),
        'strategy': strategy,
        'total_clients': requested_clients,
        'clients': []
    }
    for cid, meta in assignment_meta.items():
        entry = meta.copy()
        entry['client_id'] = cid
        manifest['clients'].append(entry)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    tasks = collect_tasks(args.source)
    assignment = assign_tasks(tasks, args.clients, args.strategy, args.mapping,
                              args.seed)
    os.makedirs(args.output, exist_ok=True)
    manifest_meta = {}
    for cid, task_list in assignment.items():
        if not task_list:
            continue
        shard_path, meta = materialize_shard(cid, task_list, args.output,
                                             args.overwrite)
        meta['shard_path'] = shard_path
        manifest_meta[cid] = meta
        print(f'Client {cid}: train={meta["num_samples"]["train"]} '
              f'samples -> {shard_path}')
    manifest_path = args.manifest or osp.join(args.output, 'manifest.json')
    write_manifest(manifest_path, 'femnist', args.source, args.strategy,
                   args.clients, manifest_meta)
    print(f'Manifest saved to {manifest_path}')


if __name__ == '__main__':
    main()
