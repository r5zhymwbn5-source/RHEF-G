"""Shared k-mer / Markov helpers for the two training-free external baselines.

This module backs ``scripts/eval_wish.py`` (WIsH) and ``scripts/eval_php.py``
(PHP).  It holds only the pieces the two runners have in common: DNA encoding,
FASTA reading, k-mer counting, the order-``k`` homogeneous Markov model that
WIsH is built from, the Gaussian log-density that PHP is built from, and the
reporting helpers.

What is faithful, and what is not
---------------------------------
Both methods are re-implementations from the authors' published source, not
wrappers around their binaries.  The places where a re-implementation can drift
are listed here once, so the runners do not have to repeat them.

**WIsH** (Galiez et al., Bioinformatics 33(19):3113-3114, 2017;
https://github.com/soedinglab/WIsH, ``mm.cpp`` / ``main.cpp``)

* order ``k = 8`` and pseudo-count ``alpha = 16.0`` are the upstream defaults
  read off ``main.cpp`` (``unsigned int order = 8;`` / ``double alpha = 16.0;``).
  The pseudo-count enters as ``(count(c,x) + alpha * 0.25) / (count(c) + alpha)``,
  i.e. a Dirichlet prior with total mass ``alpha`` spread uniformly over the four
  bases -- exactly ``mm::computeModelProb``.
* Non-ACGT characters are **dropped** and the flanks are joined
  (``mm::mapToAlphabet``), not treated as a wildcard.  Reproduced.
* Per contig the upstream code counts the ``(k+1)``-mers of the contig and then
  adds one extra pseudo-observation for the trailing ``k``-mer, so that each
  conditional distribution sums to one.  Reproduced (``wish_counts``).
* Deviation: upstream scores a phage by the **mean log-likelihood under the host
  model alone** (``mm::evaluate`` returns ``ll / length``); there is no
  background model in the code at all.  Here the score is the log-likelihood
  **ratio** against one background model, as specified for this baseline.  For a
  fixed phage the background term is a constant across hosts, so the two scores
  induce the *identical* host ranking and therefore identical predictions; only
  the softmax probabilities below differ.  The background model is fit on the
  pooled counts of **all 50 host genomes** (documented in the report); no
  leave-one-out variant is used.  Because the counts are summed array-wise, no
  artificial k-mers are created across genome boundaries, which is a (harmless,
  boundary-effect-only) difference from literally concatenating the FASTA files.
* Deviation: upstream offers an optional rejection threshold / null-model
  p-value.  That is disabled here.  The evaluation is closed-set over the 49
  genera, and ``-1`` (abstain) is reserved for a genome that yields no usable
  k-mer at all.

**PHP** (Lu et al., BMC Biology 19:12, 2021;
https://github.com/congyulu-bioinfo/PHP)

Reading the published code changes the picture relative to a prose summary of
the paper, so the actual algorithm is spelled out here:

* ``countKmer.py`` represents a genome by its 256-dimensional vector of **4-mer**
  frequencies, ``count / (len(seq) - 1)``, with 4-mers containing a non-ACGT
  character simply not counted (so the vector sums to slightly less than one on
  an assembly with ``N`` runs).  The column order is the alphabetical order of
  the 256 4-mers.  Reproduced exactly, including the ``len(seq)`` denominator.
* PHP is **not** a per-host Gaussian fitted on windows of that host's own
  genome.  ``PHP_UserTrain.py`` fits a single
  ``sklearn.mixture.GaussianMixture(n_components=1)`` -- one multivariate normal
  with a full covariance matrix -- over the space of *difference* vectors
  ``kmer(virus) - kmer(host)`` of known virus-host pairs.  One such model is
  trained per virus-length bin and shipped with the repository
  (``model/{FullLength,10k,5k,3k,1k}/*.m``, 256 x 256 covariance each).
  ``PHP.py`` then scores host ``h`` for virus ``v`` by
  ``log N(v_kmer - h_kmer | mu, Sigma)`` and predicts the argmax.
* Therefore this runner **loads the authors' five pre-trained ``.m`` files** and
  applies them unchanged (``data_raw/php_models/``, fetched from the repository).
  Nothing is re-trained: the model has seen no DatasetG genome, which is what
  keeps PHP a training-free external baseline and what rules out any leakage
  from our splits.  The length-bin cut-offs are copied from ``PHP.py``
  (``>12500`` full, ``>7500`` 10k, ``>4000`` 5k, ``>2000`` 3k, ``>4`` 1k).
* The log-density is evaluated directly from the stored ``means_`` and
  ``covariances_`` (Cholesky) rather than through ``score_samples``.  This is
  deliberate: the shipped pickles were written by scikit-learn 0.22 and are
  loaded by a much newer version, and ``score_samples`` changed its internals
  across those releases.  ``--check-sklearn`` in ``scripts/eval_php.py`` and the
  fidelity check in ``scripts/_check_external_baselines.py`` confirm that the two
  agree on the artefacts the authors shipped.
* Deviation from the paper: the paper sweeps ``k`` from 3 to 6 and reports the
  best; the released tool hard-codes ``k = 4``.  ``k = 4`` is used here because
  that is the ``k`` the shipped pre-trained models were fit on -- sweeping ``k``
  would require re-training the Gaussian, which would no longer be the published
  method.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# alphabet
# --------------------------------------------------------------------------
#: bytes -> 0 (A), 1 (C), 2 (G), 3 (T); anything else maps to -1.
_CODE = np.full(256, -1, dtype=np.int8)
for _b, _c in ((b"A", 0), (b"C", 1), (b"G", 2), (b"T", 3)):
    _CODE[_b[0]] = _c
del _b, _c

#: WIsH upstream defaults (main.cpp).
WISH_ORDER = 8
WISH_ALPHA = 16.0
#: PHP uses 4-mers in the released tool (countKmer.py, model/*.m).
PHP_K = 4


def encode(seq: str) -> np.ndarray:
    """Map a nucleotide string to an int8 array of A/C/G/T -> 0/1/2/3 codes.

    Positions that are not one of the four bases (``N``, ``R``, ``-``, ...) get
    ``-1``.  The caller decides whether to drop them (WIsH) or to only exclude
    windows that contain them (PHP).
    """
    raw = seq.upper().encode("ascii", "replace")
    return _CODE[np.frombuffer(raw, dtype=np.uint8)]


def read_fasta_records(path: Path) -> list[str]:
    """Return one upper-case sequence string per FASTA record."""
    records: list[str] = []
    chunks: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if chunks:
                    records.append("".join(chunks))
                    chunks = []
            else:
                chunks.append(line.strip())
    if chunks:
        records.append("".join(chunks))
    return records or [""]


def read_fasta_flat(path: Path) -> str:
    """Return all sequence lines concatenated, ignoring record boundaries.

    This is what PHP's ``countKmer.countFrequency`` does: it skips every ``>``
    line and joins the rest into a single string, so 4-mers *do* span contig
    boundaries in PHP.  Reproduced for fidelity.
    """
    chunks: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(">"):
                chunks.append(line.strip())
    return "".join(chunks)


# --------------------------------------------------------------------------
# k-mer counting
# --------------------------------------------------------------------------
def kmer_counts(enc: np.ndarray, k: int, chunk: int = 2_000_000) -> np.ndarray:
    """Count length-``k`` windows of ``enc`` that contain only A/C/G/T.

    ``enc`` is the output of :func:`encode`.  A window containing any ``-1`` is
    skipped rather than replaced, which is the ``if kmer4 in dic: count += 1``
    behaviour of PHP's counter.  The sequence is walked in chunks so that a
    multi-megabase genome never materialises more than a few copies of itself.
    """
    n_bins = 4 ** k
    counts = np.zeros(n_bins, dtype=np.int64)
    n = int(enc.size)
    if n < k:
        return counts
    start = 0
    while True:
        stop = min(start + chunk, n)
        seg = enc[start:stop]
        length = seg.size - k + 1
        idx = np.zeros(length, dtype=np.int64)
        valid = np.ones(length, dtype=bool)
        for j in range(k):
            sl = seg[j:j + length]
            valid &= sl >= 0
            idx *= 4
            idx += np.maximum(sl, 0)
        counts += np.bincount(idx[valid], minlength=n_bins).astype(np.int64)
        if stop >= n:
            break
        start = stop - (k - 1)  # overlap so no window is lost at the seam
    return counts


def php_feature(seq: str, k: int = PHP_K) -> np.ndarray:
    """PHP's genome descriptor: 4-mer counts divided by ``len(seq) - 1``.

    The denominator is the *raw* sequence length (including ``N``) minus one,
    not the number of windows, so the vector sums to slightly less than one on
    assemblies with ambiguous bases.  That is what ``countFrequency`` computes.
    """
    counts = kmer_counts(encode(seq), k)
    denom = max(len(seq) - 1, 1)
    return counts.astype(np.float64) / denom


# --------------------------------------------------------------------------
# WIsH: order-k homogeneous Markov model
# --------------------------------------------------------------------------
def wish_kmer_indices(enc_compact: np.ndarray, order: int = WISH_ORDER) -> np.ndarray:
    """Base-4 index of every ``(order+1)``-mer of an already-compacted sequence."""
    length = int(enc_compact.size)
    if length <= order:
        return np.empty(0, dtype=np.int64)
    n_win = length - order
    idx = np.zeros(n_win, dtype=np.int64)
    for j in range(order + 1):
        idx *= 4
        idx += enc_compact[j:j + n_win]
    return idx


def wish_counts(enc_compact: np.ndarray, order: int = WISH_ORDER):
    """Counts for one contig already reduced to A/C/G/T codes.

    Returns ``(orderCounts, lowerOrderCounts)`` with sizes ``4**(order+1)`` and
    ``4**order``, matching ``mm::countKmers``: every ``(order+1)``-mer of the
    contig is counted, its ``order``-base prefix is counted once, and the
    trailing ``order``-mer of the contig receives one extra pseudo-observation so
    that every conditional distribution sums to one.
    """
    n_hi = 4 ** (order + 1)
    n_lo = 4 ** order
    hi = np.zeros(n_hi, dtype=np.int64)
    lo = np.zeros(n_lo, dtype=np.int64)
    length = int(enc_compact.size)
    if length <= order:
        return hi, lo
    nine = wish_kmer_indices(enc_compact, order)
    hi += np.bincount(nine, minlength=n_hi).astype(np.int64)
    lo += np.bincount(nine >> 2, minlength=n_lo).astype(np.int64)
    tail = 0
    for j in range(length - order, length):
        tail = tail * 4 + int(enc_compact[j])
    lo[tail] += 1
    return hi, lo


def wish_logprobs(hi: np.ndarray, lo: np.ndarray, order: int = WISH_ORDER,
                  alpha: float = WISH_ALPHA) -> np.ndarray:
    """``log P(base | order-mer context)`` for every ``(order+1)``-mer index.

    ``hi`` may be one host's ``(4**(order+1),)`` count vector or a stack of
    hosts' ``(H, 4**(order+1))`` counts; either way the context denominators are
    gathered along the *last* axis.
    """
    context = np.arange(4 ** (order + 1), dtype=np.int64) >> 2
    lower = np.take(lo, context, axis=-1)
    return np.log((hi + alpha * 0.25) / (lower + alpha))


def compact(enc: np.ndarray) -> np.ndarray:
    """Drop non-ACGT positions, joining the flanks (WIsH's ``mapToAlphabet``)."""
    return enc[enc >= 0]


# --------------------------------------------------------------------------
# PHP: single Gaussian over virus-minus-host k-mer differences
# --------------------------------------------------------------------------
class FullGaussian:
    """One-component multivariate normal with an explicit Cholesky factor.

    ``mean``/``cov`` come straight out of the shipped ``GaussianMixture``
    pickles (``means_[0]``, ``covariances_[0]``, ``covariance_type='full'``,
    ``weights_=[1.]``).  ``logpdf`` is the textbook multivariate normal
    log-density.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        from scipy.linalg import solve_triangular

        self._solve = solve_triangular
        self.mean = np.asarray(mean, dtype=np.float64).ravel()
        self.cov = np.asarray(cov, dtype=np.float64)
        self.dim = int(self.mean.size)
        self.chol = np.linalg.cholesky(self.cov)
        self.logdet = float(2.0 * np.log(np.diag(self.chol)).sum())
        self._const = 0.5 * (self.dim * np.log(2.0 * np.pi) + self.logdet)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log-density of one vector, or of a ``(n, dim)`` batch of vectors."""
        x = np.asarray(x, dtype=np.float64)
        z = self._solve(self.chol, (x - self.mean).T, lower=True,
                        check_finite=False)
        quad = np.einsum("ij,ij->j", z, z)
        return quad * -0.5 - self._const


# --------------------------------------------------------------------------
# shared runner plumbing
# --------------------------------------------------------------------------
def softmax_over_hosts(scores: np.ndarray, host_class: np.ndarray,
                       num_classes: int) -> np.ndarray:
    """Turn a ``(n_viruses, n_hosts)`` score block into ``(n, num_classes)`` probs.

    ``host_class`` gives the DatasetG class id of each host (-1 for a host genus
    that is not in the 49-class vocabulary).  Hosts with id ``-1`` contribute no
    column, so the rows of the result sum to one only when the candidate set was
    already restricted to the vocabulary.
    """
    z = scores - scores.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    out = np.zeros((scores.shape[0], num_classes), dtype=np.float64)
    for j, c in enumerate(host_class.tolist()):
        if c >= 0:
            out[:, c] = z[:, j]
    return out.astype(np.float32, copy=False)


def macro_f1_and_acc(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro F1 over the classes the ground truth contains, plus accuracy.

    Identical in definition to ``scripts/eval_vhm.py``: ``-1`` predictions are
    counted as errors, and a class with no test row never enters the average.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present = sorted(int(c) for c in np.unique(y_true).tolist())
    per_class = []
    for c in present:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        per_class.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return {
        "macro_f1": float(np.mean(per_class)) if per_class else float("nan"),
        "accuracy": float((y_pred == y_true).mean()),
        "n_test": int(y_true.size),
        "n_classes_present": len(present),
        "n_abstained": int((y_pred < 0).sum()),
    }


def load_dataset_index(dataset: str, root: Path) -> dict:
    """Class names, labels and the ``genome_acc -> row`` map for one dataset."""
    ds = root / "features" / dataset
    info = json.loads((ds / "class_info.json").read_text(encoding="utf-8"))
    index = json.loads((ds / "genome_index.json").read_text(encoding="utf-8"))
    genomes = index["genomes"]
    return {
        "class_names": [str(n) for n in info["class_names"]],
        "num_classes": int(info["num_classes"]),
        "labels": np.load(ds / "host_labels.npy").astype(np.int64),
        "acc_to_row": {str(g["genome_acc"]): i for i, g in enumerate(genomes)},
        "n_genomes": len(genomes),
    }


def write_predictions(root: Path, dataset: str, protocol: str, method: str,
                      test_rows: np.ndarray, labels: np.ndarray,
                      y_pred: np.ndarray, y_prob: np.ndarray) -> Path:
    """Write one ``preds_<method>.npz`` in the aggregator's contract."""
    out = root / "results" / dataset / "protocols" / protocol / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"preds_{method}.npz"
    np.savez_compressed(
        path,
        y_true=labels[test_rows].astype(np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        y_prob=np.asarray(y_prob, dtype=np.float32),
    )
    return path


def write_combined_report(root: Path, dataset: str) -> Path | None:
    """Assemble ``results/aggregate/wish_php_external_<ds>.md`` from the sidecars.

    Each runner writes its own JSON next to the aggregate CSVs; this glue turns
    whatever is on disk into the single markdown table the manuscript quotes, so
    the two methods can be run (and re-run) independently without either one
    having to know about the other's progress.
    """
    agg = root / "results" / "aggregate"
    sidecars = {m: agg / f"{m.lower()}_external_{dataset}.json"
                for m in ("wish", "php")}
    loaded = {}
    for key, path in sidecars.items():
        if path.exists():
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    if not loaded:
        return None

    protocols: list[str] = []
    for blob in loaded.values():
        for p in blob.get("protocols", {}):
            if p not in protocols:
                protocols.append(p)
    protocols.sort()

    lines: list[str] = []
    lines.append(f"# Training-free external baselines: WIsH and PHP on {dataset}")
    lines.append("")
    lines.append("Both methods are re-implemented from the authors' published source and")
    lines.append("consume **no** DatasetG training split: the prediction for a genome is a")
    lines.append("function of that genome and the 50 host genomes alone. Macro F1 is averaged")
    lines.append("over the classes the ground truth of that split actually contains, exactly")
    lines.append("as in `scripts/eval_vhm.py`; `-1` (abstain) counts as an error.")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    header = "| method | " + " | ".join(f"{p} Macro F1" for p in protocols) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(protocols) + 1))
    for key, blob in loaded.items():
        cells = []
        for p in protocols:
            e = blob.get("protocols", {}).get(p)
            cells.append("—" if e is None else f"{e['macro_f1']:.4f}")
        lines.append(f"| {blob['method']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| method | " + " | ".join(f"{p} accuracy" for p in protocols) + " |")
    lines.append("|" + "---|" * (len(protocols) + 1))
    for key, blob in loaded.items():
        cells = []
        for p in protocols:
            e = blob.get("protocols", {}).get(p)
            cells.append("—" if e is None else f"{e['accuracy']:.4f}")
        lines.append(f"| {blob['method']} | " + " | ".join(cells) + " |")
    lines.append("")

    for key, blob in loaded.items():
        s = blob.get("settings", {})
        lines.append(f"## {blob['method']} — exact settings")
        lines.append("")
        for k, v in s.items():
            lines.append(f"* **{k}**: {v}")
        lines.append("")
        lines.append(f"* host generations: {len(blob.get('host_genera', []))} genomes; "
                     f"outside the 49-class vocabulary: "
                     f"{blob.get('hosts_outside_vocab') or 'none'}")
        lines.append(f"* genomes scored: {blob.get('n_scored_genomes')}")
        lines.append(f"* wall clock: {blob.get('wall_clock_sec')} s")
        lines.append("")
        lines.append(f"| protocol | n_test | Macro F1 | accuracy | abstained | classes present |")
        lines.append("|---|---|---|---|---|---|")
        for p in protocols:
            e = blob.get("protocols", {}).get(p)
            if e is None:
                continue
            lines.append(f"| {p} | {e['n_test']} | {e['macro_f1']:.4f} | "
                         f"{e['accuracy']:.4f} | {e['n_abstained']} | "
                         f"{e['n_classes_present']} |")
        lines.append("")
        if blob.get("variant_all_hosts"):
            lines.append("Unrestricted variant (all 50 host genomes as candidates; a win for")
            lines.append("`pelagibacter`, which has no class id, scores as `-1`):")
            lines.append("")
            lines.append("| protocol | Macro F1 | accuracy |")
            lines.append("|---|---|---|")
            for p in protocols:
                e = blob["variant_all_hosts"].get(p)
                if e is None:
                    continue
                lines.append(f"| {p} | {e['macro_f1']:.4f} | {e['accuracy']:.4f} |")
            lines.append("")
        if blob.get("approximations"):
            lines.append("### Approximation / deviation from the published method")
            lines.append("")
            for a in blob["approximations"]:
                lines.append(f"* {a}")
            lines.append("")

    lines.append("## Artefacts")
    lines.append("")
    for key, blob in loaded.items():
        for p in protocols:
            e = blob.get("protocols", {}).get(p)
            if e and e.get("preds_path"):
                lines.append(f"* `{e['preds_path']}`")
    lines.append("")
    lines.append("Fidelity check (PHP k-mer vectors, scores and argmax against the upstream")
    lines.append("authors' own example outputs; the WIsH model against the upstream source):")
    lines.append("`python scripts/_check_external_baselines.py`.")
    lines.append("")

    md = agg / f"wish_php_external_{dataset}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[out] {md}")
    return md

