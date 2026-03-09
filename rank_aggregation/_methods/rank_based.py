"""Rank-based aggregators."""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from statistics import fmean, median
from typing import Sequence

from rank_aggregation.base import AggregationResult, RankAggregator, Rankings
from rank_aggregation._methods.shared import chi_square_survival, kemeny_exact, kemeny_heuristic, pairwise_preferences, studentized_critical_value


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


class BordaCountAggregator(RankAggregator):
    """Borda Count: sum of per-task linear positional points."""

    @property
    def name(self) -> str:
        return "R-Borda"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        n_algorithms = len(rankings)
        scores = {
            algorithm: float(sum(n_algorithms - rank for rank in ranks))
            for algorithm, ranks in rankings.items()
        }
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class DowdallHarmonicAggregator(RankAggregator):
    """Dowdall (Harmonic) Rule: sum of reciprocal rank positions."""

    @property
    def name(self) -> str:
        return "R-Dowdall"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _score(self, ranks: Sequence[float]) -> float:
        if any(rank <= 0 for rank in ranks):
            raise ValueError("Dowdall requires strictly positive rank positions")
        return float(sum(1.0 / rank for rank in ranks))


class ReciprocalRankFusionAggregator(RankAggregator):
    """R-RRF: Reciprocal Rank Fusion with smoothing parameter k."""

    def __init__(self, k: float = 60.0) -> None:
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = float(k)

    @property
    def name(self) -> str:
        return "R-RRF"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _score(self, ranks: Sequence[float]) -> float:
        if any(rank <= 0 for rank in ranks):
            raise ValueError("RRF requires strictly positive rank positions")
        return float(sum(1.0 / (self.k + rank) for rank in ranks))


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
        p_value = chi_square_survival(chi_square, k - 1)
        if p_value > self.alpha:
            ranking = [name for name, _ in sorted(mean_ranks.items(), key=lambda item: (item[1], item[0]))]
            return AggregationResult(scores=mean_ranks, ranking=ranking)

        q_alpha = studentized_critical_value(self.alpha, k)
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
        preferences = pairwise_preferences(rankings, algorithms)

        if len(algorithms) <= self.exact_max_algorithms:
            order = kemeny_exact(algorithms, preferences)
        else:
            order = kemeny_heuristic(rankings, algorithms, preferences)

        scores = {algorithm: float(index + 1) for index, algorithm in enumerate(order)}
        return AggregationResult(scores=scores, ranking=order)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")
