"""Build provenance-tracked, train-only inputs for the new architecture.

All 1,970,436 genome/CDS occurrences are retained (maximum bag 767). DNA is a
genuine second sequence modality: canonical reverse-complement-merged 4-mers
(136 bins) and 5-mers (512 bins), each normalized by its valid-window count and
square-rooted. Invalid bases break windows; they are never deleted/joined.

PCA64 is fitted on 32,768 *unique* training CDS sampled without replacement
with seed 2026. No validation/test token is included unless that exact protein
ID is also in training. Mean ESM2 and DNA have separate training-only scalers.
The full raw 1280-dimensional genome mean remains a distinct input branch.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data.composition_evidence_data import (make_full_bags, protein_train_overlap,
                                            sha256_file, train_cds_ids)
from scripts._kmer_host_models import encode, kmer_counts

VERSION = "composition-evidence-cache-v1"
CONFIG = {"version": VERSION, "pca_dim": 64, "pca_sample": 32768,
          "seed": 2026, "pca_solver": "randomized", "pca_iterated_power": 7,
          "whiten": True, "variance_floor": 1e-12,
          "dna_k": [4, 5], "dna_transform": "sqrt(canonical_count/valid_windows)",
          "bag_limit": None, "bag_order": "source CSR; no genomic-order interpretation"}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, obj: dict):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray):
    temporary = path.with_name(path.stem + ".partial.npy")
    np.save(temporary, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays):
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def canonical_bin_map(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Full base-4 word -> sorted canonical-bin index; A/C/G/T are 0/1/2/3."""
    words = np.arange(4 ** k, dtype=np.int64)
    work = words.copy()
    reverse = np.zeros_like(words)
    for _ in range(k):
        reverse = reverse * 4 + (3 - work % 4)
        work //= 4
    canonical = np.minimum(words, reverse)
    representatives, inverse = np.unique(canonical, return_inverse=True)
    return inverse.astype(np.int32), representatives


_CANONICAL_MAPS = {k: canonical_bin_map(k) for k in (4, 5)}


def canonical_dna_counts(sequence: str, k: int) -> np.ndarray:
    mapping, representatives = _CANONICAL_MAPS.get(k, canonical_bin_map(k))
    full_counts = kmer_counts(encode(sequence), k)
    return np.bincount(mapping, weights=full_counts,
                       minlength=len(representatives)).astype(np.int64)


def dna_feature(sequence: str) -> np.ndarray:
    """648-dimensional sqrt frequencies; encode the original sequence once."""
    encoded = encode(str(sequence))
    parts = []
    for k in (4, 5):
        mapping, representatives = _CANONICAL_MAPS[k]
        full_counts = kmer_counts(encoded, k)
        counts = np.bincount(mapping, weights=full_counts,
                             minlength=len(representatives)).astype(np.float64)
        counts /= max(int(full_counts.sum()), 1)
        parts.append(np.sqrt(counts))
    return np.concatenate(parts).astype(np.float32)


def fit_scaled_train(values: np.ndarray, train: np.ndarray):
    scaler = StandardScaler().fit(np.asarray(values[train], dtype=np.float32))
    scaled = scaler.transform(np.asarray(values, dtype=np.float32)).astype(np.float32)
    return scaled, scaler


def pca_transform(values: np.ndarray, mean: np.ndarray, components: np.ndarray,
                  explained_variance: np.ndarray, variance_floor: float = 1e-12):
    """Same centered transform/whitening as sklearn, with an explicit floor."""
    transformed = np.asarray(values, dtype=np.float32) @ components.T
    transformed -= mean @ components.T
    transformed /= np.sqrt(np.maximum(explained_variance, variance_floor))
    return transformed


def source_inventory(paths: dict[str, Path], base: Path) -> dict:
    """Hash large sources once, reusing only when absolute path/stat agree."""
    inventory_path = base / "source_inventory.json"
    old = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else {}
    result = {}
    for name, path in paths.items():
        path = path.resolve()
        stat = path.stat()
        identity = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        prior = old.get(name, {})
        if all(prior.get(key) == value for key, value in identity.items()):
            identity["sha256"] = prior["sha256"]
        else:
            log(f"Hashing source {name}: {path.name} ({stat.st_size / 1e9:.2f} GB)")
            identity["sha256"] = sha256_file(path)
        result[name] = identity
    atomic_json(inventory_path, result)
    return result


def file_record(path: Path, folder: Path):
    return {"path": os.path.relpath(path, folder), "sha256": sha256_file(path),
            "bytes": path.stat().st_size}


def ensure_compatible(manifest: dict, expected: dict, path: Path):
    for key in ("config", "sources"):
        if manifest.get(key) != expected[key]:
            raise RuntimeError(f"Refusing to overwrite incompatible cache {path}; {key} differs")


def build_global(root: Path, base: Path, sources: dict, offsets, flat, n_tokens):
    global_path = base / "global_manifest.json"
    required_names = ("csr", "mean", "dna_sequences", "labels", "cds_embeddings", "generator", "loader")
    provenance = {name: sources[name] for name in required_names}
    expected = {"config": CONFIG, "sources": provenance}
    if global_path.exists():
        old = json.loads(global_path.read_text(encoding="utf-8"))
        ensure_compatible(old, expected, global_path)
        if old.get("status") == "complete":
            log("Reusing compatible complete global cache")
            return old
    bags, lengths = make_full_bags(offsets, flat, n_tokens)
    atomic_npy(base / "genome_bag_ids.npy", bags)
    atomic_npy(base / "genome_bag_lengths.npy", lengths)
    del bags
    sequences = np.load(root / "features/DatasetG/genome_sequences.npy", allow_pickle=True)
    if len(sequences) != len(lengths):
        raise ValueError("DNA sequence/genome row count mismatch")
    dna = np.empty((len(sequences), 648), dtype=np.float32)
    started = time.perf_counter()
    for row, sequence in enumerate(sequences):
        dna[row] = dna_feature(sequence)
        if row % 1000 == 0 or row + 1 == len(sequences):
            log(f"DNA rows {row + 1}/{len(sequences)}; {time.perf_counter() - started:.1f}s")
    atomic_npy(base / "dna_sqrt_features.npy", dna)
    del sequences, dna
    gc.collect()
    raw = np.load(sources["cds_embeddings"]["path"], mmap_mode="r")
    mean = np.load(root / "features/DatasetG/esm2_features.npy", mmap_mode="r")
    rng = np.random.default_rng(CONFIG["seed"])
    rows = rng.choice(len(lengths), size=min(16, len(lengths)), replace=False)
    errors = []
    for row in rows:
        ids = flat[offsets[row]:offsets[row + 1]]
        observed = np.asarray(raw[ids], dtype=np.float32).mean(axis=0)
        errors.append(float(np.abs(observed - mean[row]).max()))
    if max(errors) > 1e-5:
        raise ValueError(f"Source genome/CDS mean alignment failed: {max(errors)}")
    files = {"bag_ids": file_record(base / "genome_bag_ids.npy", base),
             "bag_lengths": file_record(base / "genome_bag_lengths.npy", base),
             "raw_dna": file_record(base / "dna_sqrt_features.npy", base)}
    manifest = {**expected, "status": "complete", "files": files,
                "dims": {"n_genomes": len(lengths), "n_tokens": n_tokens,
                         "max_bag": int(lengths.max()), "dummy_id": n_tokens,
                         "dna_dim": 648, "mean_dim": 1280, "token_dim": 64},
                "stats": {"n_cds_occurrences": int(lengths.sum()),
                          "dna_seconds": time.perf_counter() - started,
                          "mean_alignment_rows": rows.tolist(),
                          "mean_alignment_max_abs_error": max(errors)}}
    atomic_json(global_path, manifest)
    return manifest


def build_protocol(root, base, protocol, sources, global_manifest, offsets, flat):
    folder = base / protocol
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    split_path = root / "checkpoints/harp_revision/splits" / protocol / "split.npz"
    split_source = {"path": str(split_path.resolve()), "sha256": sha256_file(split_path)}
    expected = {"config": CONFIG, "sources": {**sources, "split": split_source}}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        ensure_compatible(old, expected, manifest_path)
        if old.get("status") == "complete":
            log(f"Reusing compatible complete {protocol} cache")
            return old
    started = time.perf_counter()
    split = np.load(split_path)
    train = np.asarray(split["train"], dtype=np.int64)
    train_unique = train_cds_ids(offsets, flat, train)
    rng = np.random.default_rng(CONFIG["seed"])
    sampled = rng.choice(train_unique, size=min(CONFIG["pca_sample"], len(train_unique)),
                         replace=False).astype(np.int32)
    atomic_npy(folder / "full_train_cds.npy", train_unique)
    atomic_npy(folder / "sampled_train_cds.npy", sampled)
    raw = np.load(sources["cds_embeddings"]["path"], mmap_mode="r")
    log(f"{protocol}: PCA fit on {len(sampled):,} unique train CDS; train population {len(train_unique):,}")
    fit_data = np.asarray(raw[sampled], dtype=np.float32)
    pca = PCA(n_components=CONFIG["pca_dim"], whiten=True, svd_solver="randomized",
              iterated_power=CONFIG["pca_iterated_power"], random_state=CONFIG["seed"])
    pca.fit(fit_data)
    fitted_variance = np.asarray(pca.explained_variance_, dtype=np.float32)
    if np.any(fitted_variance <= CONFIG["variance_floor"]):
        raise ValueError("PCA fitted a near-zero variance component; refusing unsafe whitening")
    check = pca_transform(fit_data[:128], pca.mean_, pca.components_, fitted_variance)
    sklearn_check = pca.transform(fit_data[:128])
    check_error = float(np.abs(check - sklearn_check).max())
    if not np.allclose(check, sklearn_check, rtol=3e-4, atol=3e-5):
        raise AssertionError(f"PCA transform differs from sklearn by {check_error}")
    atomic_npz(folder / "token_pca.npz", mean=pca.mean_, components=pca.components_,
               explained_variance=fitted_variance,
               explained_variance_ratio=pca.explained_variance_ratio_,
               singular_values=pca.singular_values_, sample_ids=sampled,
               variance_floor=np.asarray(CONFIG["variance_floor"], dtype=np.float64))
    del fit_data
    gc.collect()
    temp_tokens = folder / "token_cache.partial.npy"
    tokens = np.lib.format.open_memmap(temp_tokens, mode="w+", dtype=np.float16,
                                       shape=(len(raw) + 1, CONFIG["pca_dim"]))
    for start in range(0, len(raw), 8192):
        stop = min(start + 8192, len(raw))
        transformed = pca_transform(raw[start:stop], pca.mean_, pca.components_, fitted_variance)
        if not np.all(np.isfinite(transformed)):
            raise ValueError("Nonfinite PCA tokens")
        tokens[start:stop] = transformed.astype(np.float16)
        if start % (8192 * 20) == 0 or stop == len(raw):
            log(f"{protocol}: transformed CDS {stop:,}/{len(raw):,}")
    tokens[-1] = 0
    tokens.flush()
    del tokens
    os.replace(temp_tokens, folder / "token_cache.npy")
    raw_mean = np.load(root / "features/DatasetG/esm2_features.npy", mmap_mode="r")
    raw_dna = np.load(base / "dna_sqrt_features.npy", mmap_mode="r")
    mean_scaled, mean_scaler = fit_scaled_train(raw_mean, train)
    dna_scaled, dna_scaler = fit_scaled_train(raw_dna, train)
    atomic_npy(folder / "mean_features.npy", mean_scaled)
    atomic_npy(folder / "dna_features.npy", dna_scaled)
    atomic_npz(folder / "scalers.npz", mean_mean=mean_scaler.mean_, mean_scale=mean_scaler.scale_,
               mean_variance=mean_scaler.var_, dna_mean=dna_scaler.mean_,
               dna_scale=dna_scaler.scale_, dna_variance=dna_scaler.var_, train_genomes=train)
    overlap = protein_train_overlap(offsets, flat, train_unique, len(raw))
    atomic_npy(folder / "overlap_fraction.npy", overlap)
    atomic_npy(folder / "zero_overlap_mask.npy", overlap == 0)
    files = {name: file_record(folder / filename, folder) for name, filename in {
        "token_cache": "token_cache.npy", "token_pca": "token_pca.npz",
        "sampled_train_cds": "sampled_train_cds.npy", "full_train_cds": "full_train_cds.npy",
        "mean_features": "mean_features.npy", "dna_features": "dna_features.npy",
        "scalers": "scalers.npz", "overlap_fraction": "overlap_fraction.npy",
        "zero_overlap_mask": "zero_overlap_mask.npy"}.items()}
    files.update({name: file_record(base / filename, folder) for name, filename in {
        "bag_ids": "genome_bag_ids.npy", "bag_lengths": "genome_bag_lengths.npy",
        "raw_dna": "dna_sqrt_features.npy"}.items()})
    files["split"] = file_record(split_path, folder)
    files["labels"] = file_record(root / "features/DatasetG/host_labels.npy", folder)
    manifest = {**expected, "status": "complete", "protocol": protocol,
                "dims": global_manifest["dims"], "files": files,
                "basis_sha256": files["token_pca"]["sha256"],
                "stats": {"train_genomes": len(train), "unique_train_cds": len(train_unique),
                    "pca_fit_cds": len(sampled), "pca_transform_sklearn_max_abs_error": check_error,
                    "pca_variance_min": float(fitted_variance.min()),
                    "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
                    "test_genomes": len(split["test"]),
                    "test_zero_exact_train_protein_overlap": int((overlap[split["test"]] == 0).sum()),
                    "val_zero_exact_train_protein_overlap": int((overlap[split["val"]] == 0).sum()),
                    "build_seconds": time.perf_counter() - started}}
    atomic_json(manifest_path, manifest)
    log(f"COMPLETE {protocol}: {manifest['stats']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cds-embeddings", type=Path,
                        default=Path("D:/unihost_scratch/DatasetG_esm2_cds_fp16.npy"))
    parser.add_argument("--protocols", nargs="+", default=["Q2-canonical30", "Q3-unseen-genus"])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "features/DatasetG/architecture_revision"
    base.mkdir(parents=True, exist_ok=True)
    paths = {"cds_embeddings": args.cds_embeddings,
             "csr": root / "features/DatasetG/genome_cds_index.npz",
             "mean": root / "features/DatasetG/esm2_features.npy",
             "dna_sequences": root / "features/DatasetG/genome_sequences.npy",
             "labels": root / "features/DatasetG/host_labels.npy",
             "generator": Path(__file__),
             "loader": root / "data/composition_evidence_data.py"}
    sources = source_inventory(paths, base)
    csr = np.load(paths["csr"], allow_pickle=True)
    offsets, flat = csr["cds_offsets"], csr["cds_flat"]
    raw = np.load(args.cds_embeddings, mmap_mode="r")
    if raw.dtype != np.float16 or raw.shape[1] != 1280:
        raise ValueError("Expected raw float16 CDS embedding matrix with 1280 columns")
    with threadpool_limits(limits=args.threads):
        global_manifest = build_global(root, base, sources, offsets, flat, len(raw))
        for protocol in args.protocols:
            build_protocol(root, base, protocol, sources, global_manifest, offsets, flat)


if __name__ == "__main__":
    main()
