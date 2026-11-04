"""Composition-conditioned Distributional Evidence Pooling (CDEP).

All modes have identical module initialisation. Static and conditioned slots
differ only in the input to the query conditioner. The evidence residual is
centred against the full protein bag, so uniform attention carries no residual
information. The last residual projection starts at zero.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODES = ("mean", "dna", "mean_dna", "deepsets", "protein_slots",
         "static_slots", "conditioned_no_moments", "conditioned")
SLOT_MODES = ("protein_slots", "static_slots", "conditioned_no_moments", "conditioned")


def masked_class_probabilities(logits: np.ndarray, seen_classes) -> np.ndarray:
    """Retain the original class axis, including gaps in training labels."""
    a = np.asarray(logits, dtype=np.float64).copy()
    seen = np.asarray(seen_classes, dtype=np.int64)
    if a.ndim != 2 or len(seen) == 0 or len(np.unique(seen)) != len(seen):
        raise ValueError("Need a 2-D logit array and unique, nonempty seen classes")
    if not np.isfinite(a).all() or seen.min() < 0 or seen.max() >= a.shape[1]:
        raise ValueError("Nonfinite logits or out-of-range class IDs")
    allowed = np.zeros(a.shape[1], dtype=bool)
    allowed[seen] = True
    a[:, ~allowed] = -np.inf
    a -= a.max(axis=1, keepdims=True)
    p = np.exp(a)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


class CompositionEvidenceNet(nn.Module):
    def __init__(self, mode="conditioned", mean_dim=1280, dna_dim=648,
                 token_dim=64, num_classes=49, width=128, embedding_dim=256,
                 slots=4, dropout=0.2):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}")
        self.mode, self.width, self.slots = mode, width, slots
        self.config = dict(mode=mode, mean_dim=mean_dim, dna_dim=dna_dim,
                           token_dim=token_dim, num_classes=num_classes,
                           width=width, embedding_dim=embedding_dim,
                           slots=slots, dropout=dropout)
        self.mean_encoder = nn.Sequential(nn.Linear(mean_dim, 512), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(512, embedding_dim),
                                          nn.LayerNorm(embedding_dim))
        self.dna_encoder = nn.Sequential(nn.Linear(dna_dim, width), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(width, width))
        self.stem = nn.Sequential(nn.Linear(token_dim, width), nn.GELU(), nn.LayerNorm(width))
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.queries = nn.Parameter(torch.randn(slots, width) / math.sqrt(width))
        self.constant_condition = nn.Parameter(torch.zeros(width))
        self.query_conditioner = nn.Linear(width, slots * width, bias=False)
        self.fusion = nn.Sequential(nn.Linear(embedding_dim + width, embedding_dim),
                                    nn.GELU(), nn.LayerNorm(embedding_dim))
        self.deepsets_projection = nn.Linear(width, embedding_dim, bias=False)
        self.residual_projection = nn.Sequential(nn.Linear(slots * 2 * width, embedding_dim, bias=False),
                                                 nn.GELU(), nn.Linear(embedding_dim, embedding_dim, bias=False))
        nn.init.zeros_(self.residual_projection[-1].weight)
        self.gate = nn.Linear(embedding_dim + width, embedding_dim)
        nn.init.constant_(self.gate.bias, -2.)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    @staticmethod
    def _validate_bag(tokens, mask):
        if tokens.ndim != 3 or mask.ndim != 2 or tokens.shape[:2] != mask.shape:
            raise ValueError("tokens must be [batch, proteins, dimensions], mask [batch, proteins]")
        if mask.dtype != torch.bool:
            raise ValueError("Protein mask must be boolean")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every genome must have at least one real protein")

    def _branches(self, mean, dna):
        # Both dropout calls are made in every arm, preserving their RNG use.
        m, d = self.mean_encoder(mean), self.dna_encoder(dna)
        if self.mode == "dna":
            m = m * 0.
        if self.mode in ("mean", "protein_slots"):
            d = d * 0.
        return m, d

    def evidence(self, tokens, mask, dna_embedding, force_uniform=False):
        self._validate_bag(tokens, mask)
        # Moment calculations deliberately remain float32, including under AMP.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            h = self.stem(tokens.float())
            keys, values = self.key(h), self.value(h)
            conditional = self.mode in ("conditioned", "conditioned_no_moments")
            condition = dna_embedding.float() if conditional else self.constant_condition.unsqueeze(0).expand(len(tokens), -1)
            query = self.queries.unsqueeze(0) + self.query_conditioner(condition).reshape(-1, self.slots, self.width)
            scores = torch.einsum("bkd,bld->bkl", query, keys) / math.sqrt(self.width)
            if force_uniform:
                scores = scores * 0.
            attention = scores.masked_fill(~mask[:, None, :], -torch.inf).softmax(dim=-1)
            uniform = mask.float() / mask.sum(dim=1, keepdim=True)
            global_mean = torch.einsum("bl,bld->bd", uniform, values)
            global_var = (torch.einsum("bl,bld->bd", uniform, values.square()) - global_mean.square()).clamp_min(0.)
            means = torch.einsum("bkl,bld->bkd", attention, values)
            variances = (torch.einsum("bkl,bld->bkd", attention, values.square()) - means.square()).clamp_min(0.)
            centred_mean = means - global_mean[:, None, :]
            centred_scale = (variances + 1e-6).sqrt() - (global_var[:, None, :] + 1e-6).sqrt()
            if self.mode == "conditioned_no_moments":
                centred_scale = centred_scale * 0.
            centred = torch.cat((centred_mean, centred_scale), dim=-1).flatten(1)
            return {"attention": attention, "centred": centred, "means": means,
                    "variances": variances, "global_mean": global_mean,
                    "global_var": global_var, "stem_mean": torch.einsum("bl,bld->bd", uniform, h)}

    def inspect_evidence(self, mean, dna, tokens, mask, query_dna_override=None, force_uniform=False):
        """Expose pooling details; an override changes query context only.

        Late fusion and gating always retain the actual genome DNA descriptor.
        Static queries ignore the override exactly. Intended for eval-mode
        mechanistic perturbations; callers retain the full genome mean branch.
        """
        _, d = self._branches(mean, dna)
        if query_dna_override is not None and self.mode in ("conditioned", "conditioned_no_moments"):
            d = self.dna_encoder(query_dna_override)
        return self.evidence(tokens, mask, d, force_uniform=force_uniform)

    def encode(self, mean, dna, tokens=None, mask=None, query_dna_override=None):
        m, d = self._branches(mean, dna)
        base = self.fusion(torch.cat((m, d), dim=-1))
        if self.mode in SLOT_MODES or self.mode == "deepsets":
            self._validate_bag(tokens, mask)
            if self.mode == "deepsets":
                h = self.stem(tokens.float())
                average = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
                residual = self.deepsets_projection(average)
            else:
                query_context = d
                if query_dna_override is not None and self.mode in ("conditioned", "conditioned_no_moments"):
                    query_context = self.dna_encoder(query_dna_override)
                residual = self.residual_projection(self.evidence(tokens, mask, query_context)["centred"])
            gate = torch.sigmoid(self.gate(torch.cat((base, d), dim=-1)))
            base = base + gate * residual
        return F.normalize(self.output_norm(base), dim=-1)

    def forward(self, mean, dna, tokens=None, mask=None, query_dna_override=None):
        embedding = self.encode(mean, dna, tokens, mask, query_dna_override=query_dna_override)
        # A fixed sqrt(d) multiplier keeps the unit embedding on an MLP-like scale.
        return self.classifier(embedding * math.sqrt(embedding.shape[-1])), embedding

    def active_parameter_count(self):
        """Count structurally active parameters; no claim of equal FLOPs."""
        modules = [self.fusion, self.output_norm, self.classifier]
        if self.mode != "dna":
            modules.append(self.mean_encoder)
        if self.mode not in ("mean", "protein_slots"):
            modules.append(self.dna_encoder)
        extra = []
        if self.mode in SLOT_MODES:
            modules += [self.stem, self.key, self.value, self.query_conditioner,
                        self.residual_projection, self.gate]
            extra = [self.queries]
            if self.mode in ("protein_slots", "static_slots"):
                extra.append(self.constant_condition)
        elif self.mode == "deepsets":
            modules += [self.stem, self.deepsets_projection, self.gate]
        return sum(p.numel() for m in modules for p in m.parameters()) + sum(p.numel() for p in extra)
