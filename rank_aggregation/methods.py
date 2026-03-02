"""Aggregation methods described in README."""

from __future__ import annotations

from itertools import combinations, permutations
from math import sqrt
from statistics import NormalDist, fmean, median
from typing import Sequence

from rank_aggregation.base import AggregationResult, RankAggregator, Rankings

try:
    from scipy.stats import chi2, studentized_range
except Exception:  # pragma: no cover - optional dependency
    chi2 = None
    studentized_range = None


class MeanRankAggregator(RankAggregator):
    """R-M: arithmetic mean of ranks across tasks."""

    @property
    def name(self) -> str:
        return "R-M"

    def _score(self, ranks: Sequence[float]) -> float:
        return float(fmean(ranks))


class MedianRankAggregator(RankAggregator):
    """R-Md: median rank across tasks."""

    @property
    def name(self) -> str:
        return "R-Md"

    def _score(self, ranks: Sequence[float]) -> float:
        return float(median(ranks))


class MeanQualityAggregator(RankAggregator):
    """Q-M: arithmetic mean of quality values across tasks."""

    @property
    def name(self) -> str:
        return "Q-M"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _score(self, qualities: Sequence[float]) -> float:
        return float(fmean(qualities))


class MedianQualityAggregator(RankAggregator):
    """Q-Md: median quality value across tasks."""

    @property
    def name(self) -> str:
        return "Q-Md"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _score(self, qualities: Sequence[float]) -> float:
        return float(median(qualities))


class RescaledMeanQualityAggregator(RankAggregator):
    """Q-RM: mean of per-task min-max normalized quality values."""

    @property
    def name(self) -> str:
        return "Q-RM"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = list(rankings)
        task_count = len(next(iter(rankings.values())))

        scores: dict[str, float] = {algorithm: 0.0 for algorithm in algorithms}
        for task_idx in range(task_count):
            task_values = [rankings[algorithm][task_idx] for algorithm in algorithms]
            min_q = min(task_values)
            max_q = max(task_values)
            span = max_q - min_q
            if span == 0:
                continue
            for algorithm in algorithms:
                scores[algorithm] += (rankings[algorithm][task_idx] - min_q) / span

        for algorithm in algorithms:
            scores[algorithm] = scores[algorithm] / task_count

        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class BestRankCountAggregator(RankAggregator):
    """R-B: number of tasks where rank is 1."""

    @property
    def name(self) -> str:
        return "R-B"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _score(self, ranks: Sequence[float]) -> float:
        return float(sum(1 for rank in ranks if rank == 1))


class WorstRankCountAggregator(RankAggregator):
    """R-W: number of tasks where rank is last."""

    @property
    def name(self) -> str:
        return "R-W"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        n_algorithms = len(rankings)
        scores = {
            algorithm: float(sum(1 for rank in ranks if rank == n_algorithms))
            for algorithm, ranks in rankings.items()
        }
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class ThresholdQualityAggregator(RankAggregator):
    """Q-Th(theta): count/fraction of tasks within theta * best quality."""

    def __init__(self, theta: float = 0.95, normalize: bool = False) -> None:
        if theta <= 0 or theta > 1:
            raise ValueError("theta must be in (0, 1]")
        self.theta = theta
        self.normalize = normalize

    @property
    def name(self) -> str:
        return f"Q-Th{self.theta:g}"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = list(rankings)
        task_count = len(next(iter(rankings.values())))

        scores: dict[str, float] = {algorithm: 0.0 for algorithm in algorithms}
        for task_idx in range(task_count):
            best = max(rankings[algorithm][task_idx] for algorithm in algorithms)
            threshold = self.theta * best
            for algorithm in algorithms:
                if rankings[algorithm][task_idx] >= threshold:
                    scores[algorithm] += 1.0

        if self.normalize:
            scores = {algorithm: score / task_count for algorithm, score in scores.items()}

        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class FriedmanNemenyiRankAggregator(RankAggregator):
    """R-N: Friedman test + Nemenyi-style significance-aware ranking."""

    def __init__(self, alpha: float = 0.05) -> None:
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "R-N"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = list(rankings)
        k = len(algorithms)
        m = len(next(iter(rankings.values())))

        mean_ranks = {algorithm: float(fmean(ranks)) for algorithm, ranks in rankings.items()}
        if k < 2 or m < 2:
            ranking = sorted(algorithms)
            return AggregationResult(scores=mean_ranks, ranking=ranking)

        chi_square = ((12 * m) / (k * (k + 1))) * sum(value * value for value in mean_ranks.values())
        chi_square -= 3 * m * (k + 1)
        p_value = _chi_square_survival(chi_square, k - 1)
        if p_value > self.alpha:
            ranking = [name for name, _ in sorted(mean_ranks.items(), key=lambda item: (item[1], item[0]))]
            return AggregationResult(scores=mean_ranks, ranking=ranking)

        q_alpha = _studentized_critical_value(self.alpha, k)
        critical_difference = q_alpha * sqrt((k * (k + 1)) / (6 * m))

        wins = {algorithm: 0 for algorithm in algorithms}
        losses = {algorithm: 0 for algorithm in algorithms}
        for first, second in combinations(algorithms, 2):
            if mean_ranks[first] + critical_difference < mean_ranks[second]:
                wins[first] += 1
                losses[second] += 1
            elif mean_ranks[second] + critical_difference < mean_ranks[first]:
                wins[second] += 1
                losses[first] += 1

        ranking = sorted(
            algorithms,
            key=lambda algorithm: (-wins[algorithm], losses[algorithm], mean_ranks[algorithm], algorithm),
        )
        return AggregationResult(scores=mean_ranks, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class KemenyYoungAggregator(RankAggregator):
    """R-Kem: Kemeny-Young consensus ranking."""

    def __init__(self, exact_max_algorithms: int = 9) -> None:
        if exact_max_algorithms < 2:
            raise ValueError("exact_max_algorithms must be >= 2")
        self.exact_max_algorithms = exact_max_algorithms

    @property
    def name(self) -> str:
        return "R-Kem"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings.keys())
        preferences = _pairwise_preferences(rankings, algorithms)

        if len(algorithms) <= self.exact_max_algorithms:
            order = _kemeny_exact(algorithms, preferences)
        else:
            order = _kemeny_heuristic(rankings, algorithms, preferences)

        scores = {algorithm: float(index + 1) for index, algorithm in enumerate(order)}
        return AggregationResult(scores=scores, ranking=order)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


def _chi_square_survival(statistic: float, df: int) -> float:
    """Return P(X >= statistic) for chi-square(df)."""
    if statistic <= 0:
        return 1.0
    if chi2 is not None:  # pragma: no branch
        return float(chi2.sf(statistic, df))

    # Wilson-Hilferty normal approximation fallback.
    if df <= 0:
        return 1.0
    transformed = (statistic / df) ** (1.0 / 3.0)
    mean = 1.0 - (2.0 / (9.0 * df))
    std = sqrt(2.0 / (9.0 * df))
    z = (transformed - mean) / std
    return max(0.0, min(1.0, 1.0 - NormalDist().cdf(z)))


def _studentized_critical_value(alpha: float, k: int) -> float:
    """Critical value for Nemenyi comparisons."""
    if studentized_range is not None:  # pragma: no branch
        return float(studentized_range.isf(alpha, k, float("inf")))

    # Conservative normal approximation for pairwise family-wise control.
    comparisons = max(1, (k * (k - 1)) // 2)
    per_comparison_alpha = alpha / comparisons
    z_value = NormalDist().inv_cdf(1 - per_comparison_alpha / 2)
    return sqrt(2.0) * z_value


def _pairwise_preferences(rankings: Rankings, algorithms: list[str]) -> dict[tuple[str, str], int]:
    preferences: dict[tuple[str, str], int] = {}
    task_count = len(next(iter(rankings.values())))
    for first, second in combinations(algorithms, 2):
        first_over_second = 0
        second_over_first = 0
        for task_idx in range(task_count):
            first_rank = rankings[first][task_idx]
            second_rank = rankings[second][task_idx]
            if first_rank < second_rank:
                first_over_second += 1
            elif second_rank < first_rank:
                second_over_first += 1
        preferences[(first, second)] = first_over_second
        preferences[(second, first)] = second_over_first
    return preferences


def _kemeny_disagreement(order: Sequence[str], preferences: dict[tuple[str, str], int]) -> int:
    disagreement = 0
    for index, first in enumerate(order):
        for second in order[index + 1 :]:
            disagreement += preferences.get((second, first), 0)
    return disagreement


def _kemeny_exact(algorithms: list[str], preferences: dict[tuple[str, str], int]) -> list[str]:
    best_order: tuple[str, ...] | None = None
    best_score: int | None = None

    for candidate in permutations(algorithms):
        score = _kemeny_disagreement(candidate, preferences)
        if best_score is None or score < best_score or (score == best_score and candidate < best_order):
            best_order = candidate
            best_score = score

    return list(best_order) if best_order is not None else algorithms


def _kemeny_heuristic(
    rankings: Rankings,
    algorithms: list[str],
    preferences: dict[tuple[str, str], int],
) -> list[str]:
    # Start from Borda-style order by mean rank and improve with pair swaps.
    order = sorted(algorithms, key=lambda algorithm: (fmean(rankings[algorithm]), algorithm))
    best_score = _kemeny_disagreement(order, preferences)

    improved = True
    while improved:
        improved = False
        for left in range(len(order) - 1):
            for right in range(left + 1, len(order)):
                candidate = order[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_score = _kemeny_disagreement(candidate, preferences)
                if candidate_score < best_score:
                    order = candidate
                    best_score = candidate_score
                    improved = True
    return order

