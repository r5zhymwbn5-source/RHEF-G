"""Read immutable train-only caches for the composition/evidence architecture.

Every genome is a bag containing *all* of its CDS identifiers. Token order is
the source CSR order and carries no genomic-order interpretation. Cached tokens
are PCA-whitened ESM2 CDS embeddings; the other two branches are independently
train-standardized genome mean ESM2 and true canonical DNA 4/5-mer descriptors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_cache(root: str | Path, protocol: str, verify_hashes: bool = False) -> dict:
    """Return paths, dimensions, manifest, and split for a completed cache.

    Arrays remain on disk until the caller explicitly loads them. Global bag
    IDs are int32; convert to int64 for PyTorch indexing. ``token_cache`` has a
    final exactly-zero dummy row; padding IDs refer only to that row.
    """
    root = Path(root).resolve()
    base = root / "features" / "DatasetG" / "architecture_revision"
    folder = base / protocol
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Cache is not complete: {manifest_path}")
    result = {name: (folder / value["path"]).resolve()
              for name, value in manifest["files"].items()}
    for name, path in result.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        if verify_hashes and sha256_file(path) != manifest["files"][name]["sha256"]:
            raise RuntimeError(f"Cache hash mismatch for {name}: {path}")
    result.update(root=root, protocol=protocol, folder=folder,
                  manifest_path=manifest_path, manifest=manifest,
                  dims=manifest["dims"], dummy_id=manifest["dims"]["dummy_id"])
    return result


def make_full_bags(offsets: np.ndarray, flat: np.ndarray,
                   dummy_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Preserve every source row's CDS, including repeated IDs, without caps."""
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(flat):
        raise ValueError("Invalid CSR offsets")
    if np.any(lengths < 1) or np.any(flat < 0) or np.any(flat >= dummy_id):
        raise ValueError("Genome bags must contain valid CDS IDs")
    bags = np.full((len(lengths), int(lengths.max())), dummy_id, dtype=np.int32)
    for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        bags[row, :stop - start] = flat[start:stop]
    return bags, lengths.astype(np.int32)


def train_cds_ids(offsets: np.ndarray, flat: np.ndarray,
                  train_genomes: np.ndarray) -> np.ndarray:
    """Unique CDS IDs from training genomes, without label-based filtering."""
    chunks = [flat[offsets[row]:offsets[row + 1]] for row in train_genomes]
    return np.unique(np.concatenate(chunks)).astype(np.int32)


def protein_train_overlap(offsets: np.ndarray, flat: np.ndarray,
                          train_ids: np.ndarray, n_tokens: int) -> np.ndarray:
    """Fraction of a genome's CDS occurrences with exact IDs present in train.

    A zero is an exact-protein-ID diagnostic, not a remote homology guarantee.
    """
    seen = np.zeros(n_tokens, dtype=bool)
    seen[train_ids] = True
    hits = seen[flat].astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(hits)))
    return ((prefix[offsets[1:]] - prefix[offsets[:-1]]) /
            np.diff(offsets)).astype(np.float32)
