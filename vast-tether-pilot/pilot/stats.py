"""Statistics for the early-evidence pilot: stratified AUROC with a cluster bootstrap, and belief pooling."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The position bins of the paper's Table decay (rbd_common.BINS).
POSITION_BINS = ((0, 15), (16, 31), (32, 63), (64, 95), (96, 127), (128, 191), (192, 255), (256, 383), (384, 512))


def position_bin(t: int) -> int:
    for index, (low, high) in enumerate(POSITION_BINS):
        if low <= t <= high:
            return index
    raise ValueError(f"position {t} outside the bins")


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(positive > negative) + 0.5 P(tie)."""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    difference = positive[:, None] - negative[None, :]
    return float(((difference > 0) + 0.5 * (difference == 0)).mean())


@dataclass
class _Stratum:
    wins: np.ndarray
    positive_clusters: np.ndarray
    negative_clusters: np.ndarray


class StratifiedAuroc:
    """AUROC computed inside strata and averaged, with weights per cluster (image) for a paired bootstrap.

    Strata that lack either class contribute nothing. Strata are weighted by their number of pairs, as a stratified
    Mann-Whitney statistic is, unless equal_strata asks for the plain mean over strata (used for per-object AUROCs).
    """

    def __init__(self, scores: np.ndarray, positive: np.ndarray, strata: np.ndarray, clusters: np.ndarray,
                 equal_strata: bool = False) -> None:
        keep = ~np.isnan(scores)
        scores, positive, strata, clusters = scores[keep], positive[keep], strata[keep], clusters[keep]
        self.equal_strata = equal_strata
        self.strata: list[_Stratum] = []
        self.n_positive = int(positive.sum())
        self.n_negative = int((~positive).sum())
        for stratum in np.unique(strata):
            inside = strata == stratum
            pos, neg = inside & positive, inside & ~positive
            if pos.any() and neg.any():
                difference = scores[pos][:, None] - scores[neg][None, :]
                self.strata.append(_Stratum((difference > 0) + 0.5 * (difference == 0), clusters[pos], clusters[neg]))

    def value(self, cluster_weights: np.ndarray | None = None) -> float:
        numerator, denominator, per_stratum = 0.0, 0.0, []
        for stratum in self.strata:
            if cluster_weights is None:
                wins = float(stratum.wins.sum())
                pairs = float(stratum.wins.size)
            else:
                wp, wn = cluster_weights[stratum.positive_clusters], cluster_weights[stratum.negative_clusters]
                wins = float(wp @ stratum.wins @ wn)
                pairs = float(wp.sum() * wn.sum())
            if pairs > 0:
                numerator += wins
                denominator += pairs
                per_stratum.append(wins / pairs)
        if self.equal_strata:
            return float(np.mean(per_stratum)) if per_stratum else float("nan")
        return numerator / denominator if denominator else float("nan")


@dataclass
class PairedBootstrap:
    """Percentile intervals over images. One set of cluster weights is shared by every estimator, so differences pair."""

    n_clusters: int
    n_resamples: int = 2000
    seed: int = 0
    _weights: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._weights = [np.bincount(rng.integers(0, self.n_clusters, self.n_clusters), minlength=self.n_clusters).astype(float)
                         for _ in range(self.n_resamples)]

    def ci(self, estimator: StratifiedAuroc) -> tuple[float, float]:
        draws = np.array([estimator.value(w) for w in self._weights])
        return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))

    def difference_ci(self, first: StratifiedAuroc, second: StratifiedAuroc) -> tuple[float, float]:
        draws = np.array([first.value(w) - second.value(w) for w in self._weights])
        return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def pooled_belief(evidence: np.ndarray, t: int, weights: np.ndarray, include_writing_step: bool = True,
                  mask: np.ndarray | None = None) -> float:
    """Weighted mean of evidence over steps u <= t (or u < t). Matches A(t)/R(t) of vdd_decode.Evidence."""
    end = t + 1 if include_writing_step else t
    step_weights = weights[:end].astype(float)
    if mask is not None:
        step_weights = step_weights * mask[:end]
    total = step_weights.sum()
    if end <= 0 or total <= 0:
        return float("nan")
    return float(step_weights @ evidence[:end] / total)


def schedule_from_kl(bin_means: dict[tuple[int, int], float], length: int) -> np.ndarray:
    """r(t) of rbd_common.r_curve: gray-KL by bin, normalized to 1 at the first bin, interpolated between bin centres."""
    bins = sorted(bin_means)
    centres = [(low + high) / 2 for low, high in bins]
    values = np.array([bin_means[b] for b in bins], dtype=float)
    return np.interp(np.arange(length), centres, values / values[0])


def robust_z(value: np.ndarray, median: np.ndarray, q25: np.ndarray, q75: np.ndarray) -> np.ndarray:
    """Centre on the step's median over V_obj and scale by its interquartile range."""
    return (value - median) / np.maximum(q75 - q25, 1e-6)
