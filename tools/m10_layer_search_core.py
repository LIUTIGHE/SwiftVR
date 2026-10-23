"""Deterministic mask proposals and disjoint data splits for M10 layer search.

No model, cache, or inference implementation is replaced here. Search produces
static ordered layer subsets; recovered candidates are ranked separately.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import random


def checked_mask(indices, total=30, keep=None, protect_edges=True):
    mask = tuple(int(i) for i in indices)
    if not mask or tuple(sorted(set(mask))) != mask:
        raise ValueError("Layer indices must be sorted, unique and non-empty")
    if mask[0] < 0 or mask[-1] >= total or (keep is not None and len(mask) != keep):
        raise ValueError("Invalid layer count or source index")
    if protect_edges and (mask[0] != 0 or mask[-1] != total - 1):
        raise ValueError("This gate preserves the first and last source blocks")
    return mask


def mask_id(mask):
    payload = ",".join(map(str, mask)).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def source_disjoint_split(samples, validation_samples, *, probe=8, rank=16,
                          adapt=128, seed=0):
    """Group ALL variants/views by sample_id; never split views of a source.

    Equal sample_id strings in different datasets are conservatively treated as
    one source. This cannot detect external aliases of the same source video.
    """
    if min(probe, rank, adapt) <= 0:
        raise ValueError("Split sizes must be positive")
    for s in [*samples, *validation_samples]:
        if not isinstance(s.get("sample_id"), str) or not s["sample_id"]:
            raise ValueError("sample_id is required for source-disjoint splitting")
    excluded = {s["sample_id"] for s in validation_samples}
    groups = {}
    seen = set()
    for s in samples:
        idx = int(s["distillation_index"])
        if idx in seen:
            raise ValueError("Duplicate cache index")
        seen.add(idx)
        if s["sample_id"] not in excluded:
            groups.setdefault(s["sample_id"], []).append(idx)
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    if len(names) <= probe + rank:
        raise ValueError("Not enough distinct TRAIN sources for probe/rank/adapt")
    pnames, rnames = names[:probe], names[probe:probe + rank]
    anames = names[probe + rank:]
    for name in sorted(groups):
        indices = groups[name]
        indices.sort()
        rng.shuffle(indices)
    # Round robin prioritizes diversity before taking extra views of a source.
    available = [groups[n][j] for j in range(max(map(len, groups.values())))
                 for n in anames if j < len(groups[n])]
    if len(available) < adapt:
        raise ValueError(f"Need {adapt} adaptation views, only {len(available)} available")
    aindices = available[:adapt]
    used = set(aindices)
    return {
        "probe": [groups[n][0] for n in pnames],
        "rank": [groups[n][0] for n in rnames],
        "adapt": aindices,
        "source_groups": {"probe": pnames, "rank": rnames,
                          "adapt": [n for n in anames if used.intersection(groups[n])]},
        "excluded_validation_sources": sorted(excluded),
        "grouping": "sample_id across variants/views; external aliases not detected",
    }


def propose_swaps(mask, drop_scores, add_scores, *, width=3, total=30):
    """Use neighboring-depth probes only to propose; joint L20 is re-evaluated."""
    mask = checked_mask(mask, total=total)
    if width <= 0:
        raise ValueError("Proposal width must be positive")
    if any(not math.isfinite(float(v)) for v in [*drop_scores.values(), *add_scores.values()]):
        raise ValueError("Non-finite proposal score")
    removals = sorted(drop_scores, key=lambda i: (drop_scores[i], i))[:width]
    additions = sorted(add_scores, key=lambda i: (add_scores[i], i))[:width]
    results = set()
    for remove in removals:
        if remove not in mask or remove in (0, total - 1):
            raise ValueError("Invalid removal")
        for add in additions:
            if add in mask or not 0 < add < total - 1:
                raise ValueError("Invalid addition")
            candidate = tuple(sorted((set(mask) - {remove}) | {add}))
            results.add(checked_mask(candidate, total=total, keep=len(mask)))
    return sorted(results)


def shortlist(scored, baseline, count=4):
    """Always retain the historical heuristic as a same-budget control."""
    if count < 2 or baseline not in scored:
        raise ValueError("Need baseline and at least two candidate slots")
    if any(not math.isfinite(float(v)) for v in scored.values()):
        raise ValueError("Non-finite screen score")
    others = sorted((m for m in scored if m != baseline), key=lambda m: (scored[m], m))
    return [baseline, *others[:count - 1]]


def training_order(size, updates, accumulation, seed):
    if min(size, updates, accumulation) <= 0:
        raise ValueError("Training sizes must be positive")
    rng, result = random.Random(seed), []
    while len(result) < updates * accumulation:
        values = list(range(size))
        rng.shuffle(values)
        result.extend(values)
    return result[:updates * accumulation]


def recovered_winner(records):
    """Selection uses search-heldout LPIPS, not train loss or official val13."""
    if not records:
        raise ValueError("No recovered candidates")
    if len({r["steps"] for r in records}) != 1:
        raise ValueError("Cannot rank unequal recovery budgets")
    for r in records:
        if not all(math.isfinite(float(r[k])) for k in ("lpips", "rgb_mse")):
            raise ValueError("Non-finite recovery score")
    return min(records, key=lambda r: (r["lpips"], r["rgb_mse"], r["id"]))


@contextmanager
def compact_block_view(model, kept):
    """Actually execute only kept blocks, with EXPORTED L20 window parity.

    Only no-grad screening uses this view. Recovery uses a separately constructed
    model. Restore original module list and every shift flag even on exceptions.
    """
    import torch.nn as nn
    original = model.blocks
    mask = checked_mask(kept, total=len(original), protect_edges=False)
    if any(not hasattr(b, "attn1") or not hasattr(b.attn1, "_do_shift") for b in original):
        raise ValueError("Prepare the training-safe attention processor first")
    shifts = [b.attn1._do_shift for b in original]
    try:
        model.blocks = nn.ModuleList([original[i] for i in mask])
        for i, block in enumerate(model.blocks):
            block.attn1._do_shift = bool(i % 2)
        yield model
    finally:
        model.blocks = original
        for block, shift in zip(original, shifts):
            block.attn1._do_shift = shift
