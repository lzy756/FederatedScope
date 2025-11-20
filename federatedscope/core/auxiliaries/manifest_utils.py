import json
import logging
import os.path as osp
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_manifest(path: str, strict: bool = True) -> Optional[Dict[str, Any]]:
    """Load and parse a manifest JSON file if available."""
    if not path:
        return None
    abs_path = osp.abspath(path)
    if not osp.exists(abs_path):
        message = f"Manifest file {abs_path} not found"
        if strict:
            raise FileNotFoundError(message)
        logger.warning(message + "; skip manifest validation.")
        return None
    try:
        with open(abs_path, 'r') as f:
            manifest = json.load(f)
        return manifest
    except json.JSONDecodeError as err:
        message = f"Failed to parse manifest {abs_path}: {err}"
        if strict:
            raise ValueError(message) from err
        logger.warning(message)
        return None


def find_client_in_manifest(
        manifest: Optional[Dict[str, Any]],
        client_id: int,
        shard_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the manifest entry for the given client id if possible."""
    if not manifest:
        return None
    entries = manifest.get('clients', [])
    abs_shard = osp.abspath(shard_path) if shard_path else None
    for entry in entries:
        entry_id = entry.get('client_id', entry.get('id', -1))
        if entry_id != client_id:
            continue
        manifest_path = entry.get('shard_path')
        if abs_shard and manifest_path:
            if osp.abspath(manifest_path) != abs_shard:
                continue
        return entry
    return None


def summarize_manifest(manifest: Optional[Dict[str, Any]]) -> str:
    if not manifest:
        return "<empty>"
    dataset = manifest.get('dataset', 'unknown')
    clients = manifest.get('clients', [])
    return f"dataset={dataset}, clients={len(clients)}"
