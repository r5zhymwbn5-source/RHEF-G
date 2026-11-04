"""Fixed-budget, train-only-preprocessed CDEP ablations on full protein bags.

Run training first with --defer-test; evaluate the locked complete grid with
--evaluate-only. Validation Macro F1 alone chooses the checkpoint. Test exports
are never produced by a deferred training run. Every arm has the same uniform
genome-row draws and cross entropy objective, with no early stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.composition_evidence_data import load_cache, sha256_file
from models.composition_evidence import MODES, SLOT_MODES, CompositionEvidenceNet, masked_class_probabilities

VERSION = "cdep-1.0-fullbag-matched-uniform-ce"


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def metric_scores(y, prob):
    pred, present = prob.argmax(axis=1), np.unique(y)
    ap = [average_precision_score(y == c, prob[:, c]) for c in present if 0 < np.sum(y == c) < len(y)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = balanced_accuracy_score(y, pred)
    return {"macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
            "macro_f1_union": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced),
            "mcc": float(matthews_corrcoef(y, pred)), "macro_auprc": float(np.mean(ap)) if ap else None,
            "n_rows": len(y), "n_present_classes": len(present),
            "macro_f1_label_policy": "classes present in true labels"}


class DeviceCache:
    def __init__(self, cache, device):
        self.device = torch.device(device)
        self.dummy_id = cache["dummy_id"]
        self.mean = torch.as_tensor(np.load(cache["mean_features"]), dtype=torch.float32, device=device)
        self.dna = torch.as_tensor(np.load(cache["dna_features"]), dtype=torch.float32, device=device)
        self.tokens = torch.as_tensor(np.load(cache["token_cache"]), dtype=torch.float16, device=device)
        self.ids = torch.as_tensor(np.load(cache["bag_ids"]).astype(np.int64), device=device)
        lengths = np.load(cache["bag_lengths"]).astype(np.int64)
        self.lengths_numpy = lengths
        self.lengths = torch.as_tensor(lengths, device=device)
        self.y_numpy = np.load(cache["labels"]).astype(np.int64)
        self.targets = torch.as_tensor(self.y_numpy, device=device)
        self.num_classes = max(49, int(self.y_numpy.max()) + 1)
        n = len(self.y_numpy)
        if not (len(self.mean) == len(self.dna) == len(self.ids) == len(lengths) == n):
            raise ValueError("Cache genome-axis mismatch")
        if (lengths < 1).any() or lengths.max() != self.ids.shape[1]:
            raise ValueError("Invalid or unexpectedly capped protein bags")
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.dna).all() and torch.isfinite(self.tokens).all()):
            raise ValueError("Nonfinite cache values")
        with np.load(cache["split"]) as z:
            self.split = {k: np.asarray(z[k], dtype=np.int64) for k in ("train", "val", "test")}
        joined = np.concatenate(list(self.split.values()))
        if len(np.unique(joined)) != len(joined) or joined.min() < 0 or joined.max() >= n:
            raise ValueError("Invalid split or partition overlap")
        self.seen = np.unique(self.y_numpy[self.split["train"]])

    def batch(self, rows, mode):
        ix = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        if mode in SLOT_MODES or mode == "deepsets":
            max_length = int(self.lengths_numpy[rows].max())
            ids = self.ids[ix, :max_length]
            tokens = self.tokens[ids].float()
            mask = ids != self.dummy_id
        else:
            tokens = mask = None
        return self.mean[ix], self.dna[ix], tokens, mask


@torch.no_grad()
def infer(model, data, indices, batch_size=128):
    model.eval()
    logits, embeddings = [], []
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        a, b = model(*data.batch(rows, model.mode))
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise FloatingPointError("Nonfinite inference output")
        logits.append(a.cpu().numpy())
        embeddings.append(b.cpu().numpy())
    return np.concatenate(logits), np.concatenate(embeddings)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def sources(cache):
    paths = {"runner": Path(__file__), "model": ROOT / "models/composition_evidence.py",
             "cache_loader": ROOT / "data/composition_evidence_data.py", "cache_manifest": cache["manifest_path"],
             "cache_generator": ROOT / "scripts/build_composition_evidence_cache.py",
             "dna_encoder_helper": ROOT / "scripts/_kmer_host_models.py",
             "architecture_tests": ROOT / "tests/test_composition_evidence.py"}
    return {k: {"path": str(p.resolve()), "sha256": sha256_file(p)} for k, p in paths.items()}


def snapshot_sources(root, src):
    folder = root / "source_snapshot"
    folder.mkdir(parents=True, exist_ok=True)
    for key, item in src.items():
        # Protocol manifests live inside their own run manifest, not one common filename.
        if key == "cache_manifest":
            continue
        target = folder / Path(item["path"]).name
        if target.exists() and sha256_file(target) != item["sha256"]:
            raise ValueError(f"Source changed after experiment lock: {target}")
        if not target.exists():
            shutil.copyfile(item["path"], target)


def run_one(args, protocol, seed, mode, data, cache, src):
    start = time.perf_counter()
    model_config = dict(mode=mode, mean_dim=data.mean.shape[1], dna_dim=data.dna.shape[1],
                        token_dim=data.tokens.shape[1], num_classes=data.num_classes,
                        width=128, embedding_dim=256, slots=4, dropout=0.2)
    cfg = {"version": VERSION, "protocol": protocol, "seed": seed, "mode": mode,
           "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size,
           "lr": args.lr, "weight_decay": args.weight_decay, "clip_grad_norm": 1.,
           "model": model_config, "sources": src, "cache_files": cache["manifest"]["files"],
           "sampling": "uniform genome rows without replacement independently at every step",
           "objective": "unweighted cross entropy", "precision": "float32 computation; float16 cached PCA tokens",
           "selection": "validation Macro F1 on true-present classes; earliest best tie",
           "schedule": "constant learning rate", "full_bags": True, "test_used_for_selection": False}
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    out, ckpt = args.results_root / protocol / f"seed{seed}" / mode, args.checkpoint_root / protocol / f"seed{seed}" / mode
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_digest"] != digest:
            raise ValueError(f"Existing run configuration/source mismatch: {out}")
        if manifest["status"] == "complete" or (manifest["status"] == "trained" and args.defer_test):
            print(f"SKIP {protocol} seed{seed} {mode} {manifest['status']}", flush=True)
            return manifest
    if args.evaluate_only and (manifest is None or manifest["status"] not in ("trained", "complete")):
        raise ValueError(f"Evaluation requires completed fixed-budget training: {out}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if data.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = CompositionEvidenceNet(**model_config).to(data.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    row_rng = np.random.default_rng(seed)
    history, start_epoch, updates, exposures, cds_positions, padded_positions = [], 0, 0, 0, 0, 0
    best_score, best_epoch, previous_seconds = -1., 0, 0.
    row_chain = bytes(32)
    last_path = ckpt / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=data.device, weights_only=False)
        if state["config_digest"] != digest:
            raise ValueError("Resume source/config mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, start_epoch = state["history"], state["epoch"]
        updates, exposures = state["optimizer_updates"], state["sample_exposures"]
        cds_positions, padded_positions = state["cds_positions"], state["padded_positions"]
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        previous_seconds, row_chain = state["runtime_seconds"], bytes.fromhex(state["row_chain"])
        row_rng.bit_generator.state = state["row_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if data.device.type == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
    if manifest is None:
        manifest = {"status": "running", "config_digest": digest, "config": cfg,
                    "split_sizes": {k: len(v) for k, v in data.split.items()}, "train_classes": data.seen.tolist(),
                    "device": str(data.device), "torch_version": torch.__version__, "numpy_version": np.__version__,
                    "gpu": torch.cuda.get_device_name(data.device) if data.device.type == "cuda" else None,
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                    "active_parameters": model.active_parameter_count(), "parameter_note": "All modules instantiated in all arms; only structurally active branches counted separately. Equal steps and genome rows, unequal FLOPs.",
                    "test_exported": False}
        dump_json(manifest_path, manifest)
    dump_json(ckpt / "model_config.json", model_config)
    print(f"START {protocol} seed{seed} {mode} epoch={start_epoch}/{args.epochs} active={manifest['active_parameters']}", flush=True)
    if not args.evaluate_only:
        train = data.split["train"]
        for epoch in range(start_epoch, args.epochs):
            synchronize(data.device)
            epoch_start = time.perf_counter()
            model.train()
            losses, gradient_norms = [], []
            batch_hasher = hashlib.sha256()
            for _ in range(args.steps_per_epoch):
                rows = row_rng.choice(train, args.batch_size, replace=False).astype(np.int64)
                batch_hasher.update(rows.tobytes())
                row_chain = hashlib.sha256(row_chain + rows.tobytes()).digest()
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(*data.batch(rows, mode))
                loss = F.cross_entropy(logits, data.targets[torch.as_tensor(rows, device=data.device)])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss {protocol}/{seed}/{mode}/{epoch}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                gradient_norms.append(float(grad_norm))
                updates += 1
                exposures += len(rows)
                if mode in SLOT_MODES or mode == "deepsets":
                    cds_positions += int(data.lengths_numpy[rows].sum())
                    padded_positions += len(rows) * int(data.lengths_numpy[rows].max())
            synchronize(data.device)
            train_seconds = time.perf_counter() - epoch_start
            val_start = time.perf_counter()
            val_logits, _ = infer(model, data, data.split["val"], args.eval_batch_size)
            prob = masked_class_probabilities(val_logits, data.seen)
            score = float(f1_score(data.y_numpy[data.split["val"]], prob.argmax(axis=1),
                                   labels=np.unique(data.y_numpy[data.split["val"]]), average="macro", zero_division=0))
            synchronize(data.device)
            if score > best_score + 1e-12:
                best_score, best_epoch = score, epoch + 1
                torch.save({"model": model.state_dict(), "model_config": model_config, "config_digest": digest,
                            "epoch": best_epoch, "val_macro_f1": best_score}, ckpt / "best.pt")
            record = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "gradient_norm": float(np.mean(gradient_norms)),
                      "val_macro_f1": score, "train_seconds": train_seconds,
                      "validation_seconds": time.perf_counter() - val_start,
                      "optimizer_updates": updates, "sample_exposures": exposures,
                      "cds_positions": cds_positions, "padded_positions": padded_positions,
                      "batch_rows_sha256": batch_hasher.hexdigest(), "cumulative_rows_sha256": row_chain.hex()}
            history.append(record)
            elapsed = previous_seconds + time.perf_counter() - start
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config_digest": digest,
                        "epoch": epoch + 1, "history": history, "best_score": best_score, "best_epoch": best_epoch,
                        "optimizer_updates": updates, "sample_exposures": exposures, "cds_positions": cds_positions,
                        "padded_positions": padded_positions, "runtime_seconds": elapsed,
                        "row_rng": row_rng.bit_generator.state, "row_chain": row_chain.hex(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if data.device.type == "cuda" else []}, last_path)
            dump_json(out / "history.json", history)
            manifest.update(last_epoch=epoch + 1, best_epoch=best_epoch, best_val_macro_f1=best_score,
                            optimizer_updates=updates, sample_exposures=exposures, cds_positions=cds_positions,
                            padded_positions=padded_positions, cumulative_rows_sha256=row_chain.hex(), runtime_seconds=elapsed)
            dump_json(manifest_path, manifest)
            print(f"{protocol} seed{seed} {mode} ep={epoch+1}/{args.epochs} loss={record['loss']:.4f} "
                  f"valF1={score:.4f} best={best_score:.4f}@{best_epoch} train_s={train_seconds:.2f} "
                  f"val_s={record['validation_seconds']:.2f}", flush=True)
    if len(history) != args.epochs or updates != args.epochs * args.steps_per_epoch:
        raise RuntimeError("Run did not reach the fixed optimization budget")
    best = torch.load(ckpt / "best.pt", map_location=data.device, weights_only=False)
    model.load_state_dict(best["model"])
    results = {}
    for partition in (("val",) if args.defer_test else ("val", "test")):
        indices = data.split[partition]
        logits, embedding = infer(model, data, indices, args.eval_batch_size)
        probs = masked_class_probabilities(logits, data.seen)
        results[partition] = metric_scores(data.y_numpy[indices], probs)
        np.savez_compressed(out / f"{partition}_predictions.npz", indices=indices, y_true=data.y_numpy[indices],
                            probs=probs, embedding=embedding, logits=logits,
                            class_ids=np.arange(data.num_classes), seen_classes=data.seen)
    manifest.update(status="trained" if args.defer_test else "complete", best_epoch=best_epoch,
                    best_val_macro_f1=best_score, optimizer_updates=updates, sample_exposures=exposures,
                    cds_positions=cds_positions, padded_positions=padded_positions,
                    cumulative_rows_sha256=row_chain.hex(), results=results, test_exported=not args.defer_test,
                    runtime_seconds=previous_seconds + time.perf_counter() - start,
                    training_seconds=sum(v["train_seconds"] for v in history),
                    validation_seconds=sum(v["validation_seconds"] for v in history))
    dump_json(manifest_path, manifest)
    print(f"DONE {protocol} seed{seed} {mode} status={manifest['status']} val={results['val']['macro_f1']:.4f}", flush=True)
    del model, optimizer
    if data.device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", default="Q2-canonical30,Q3-unseen-genus")
    parser.add_argument("--seeds", default="42,2024,777")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps-per-epoch", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/architecture_revision")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/architecture_revision")
    parser.add_argument("--defer-test", "--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only and args.defer_test:
        parser.error("--evaluate-only and --defer-test are exclusive")
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.eval_batch_size) < 1:
        parser.error("Training and evaluation budgets must be positive")
    modes = args.modes.split(",")
    if not set(modes).issubset(MODES) or len(set(modes)) != len(modes):
        parser.error("Unknown or duplicate modes")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    summaries = []
    for protocol in args.protocols.split(","):
        cache = load_cache(ROOT, protocol, verify_hashes=True)
        src = sources(cache)
        snapshot_sources(args.results_root, src)
        data = DeviceCache(cache, args.device)
        print(f"CACHE {protocol} tokens={tuple(data.tokens.shape)} fullbag_max={data.ids.shape[1]} "
              f"gpu_MB={torch.cuda.memory_allocated()/2**20 if data.device.type == 'cuda' else 0:.1f}", flush=True)
        for seed in map(int, args.seeds.split(",")):
            for mode in modes:
                summaries.append(run_one(args, protocol, seed, mode, data, cache, src))
        del data
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    dump_json(args.results_root / ("evaluation_grid_summary.json" if args.evaluate_only else "training_grid_summary.json"), summaries)


if __name__ == "__main__":
    main()
