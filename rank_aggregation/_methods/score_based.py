"""Score-based and benchmark-profile aggregators."""

from __future__ import annotations

from math import exp, log
from statistics import fmean, median
from typing import Sequence

from rank_aggregation.base import AggregationResult, RankAggregator, Rankings
from rank_aggregation._methods.shared import dm_auc_scores


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


class GeometricMeanQualityAggregator(RankAggregator):
    """Q-GM: geometric mean of quality values across tasks."""

    @property
    def name(self) -> str:
        return "Q-GM"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _score(self, qualities: Sequence[float]) -> float:
        if any(value < 0 for value in qualities):
            raise ValueError("Q-GM requires non-negative quality values")
        if any(value == 0 for value in qualities):
            return 0.0
        return float(exp(fmean(log(value) for value in qualities)))


class HarmonicMeanQualityAggregator(RankAggregator):
    """Q-HM: harmonic mean of quality values across tasks."""

    @property
    def name(self) -> str:
        return "Q-HM"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _score(self, qualities: Sequence[float]) -> float:
        if any(value <= 0 for value in qualities):
            raise ValueError("Q-HM requires strictly positive quality values")
        count = len(qualities)
        return float(count / sum(1.0 / value for value in qualities))


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
            scores[algorithm] /= task_count

        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
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


class DMAUCPerformanceProfileAggregator(RankAggregator):
    """DM-AUC: Dolan-More performance-profile AUC ranking."""

    def __init__(self, tau_max: float | None = None, normalize_auc: bool = True) -> None:
        if tau_max is not None and tau_max <= 1.0:
            raise ValueError("tau_max must be > 1 when provided")
        self.tau_max = tau_max
        self.normalize_auc = normalize_auc

    @property
    def name(self) -> str:
        return "DM-AUC"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        if any(value <= 0 for values in rankings.values() for value in values):
            raise ValueError("DM-AUC requires strictly positive per-task values")

        algorithms = sorted(rankings)
        scores = dm_auc_scores(
            rankings,
            algorithms,
            tau_max=self.tau_max,
            normalize_auc=self.normalize_auc,
        )
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class DMLBOLeaveOneOutProfileAggregator(RankAggregator):
    """DM-LBO: iterative leave-one-out performance-profile ranking."""

    def __init__(self, tau_max: float | None = None, normalize_auc: bool = True) -> None:
        if tau_max is not None and tau_max <= 1.0:
            raise ValueError("tau_max must be > 1 when provided")
        self.tau_max = tau_max
        self.normalize_auc = normalize_auc

    @property
    def name(self) -> str:
        return "DM-LBO"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        if any(value <= 0 for values in rankings.values() for value in values):
            raise ValueError("DM-LBO requires strictly positive per-task values")

        total_algorithms = len(rankings)
        remaining = sorted(rankings)
        ranking: list[str] = []
        while remaining:
            auc_scores = dm_auc_scores(
                rankings,
                remaining,
                tau_max=self.tau_max,
                normalize_auc=self.normalize_auc,
            )
            winner = min(remaining, key=lambda algorithm: (-auc_scores[algorithm], algorithm))
            ranking.append(winner)
            remaining.remove(winner)

        scores = {algorithm: float(total_algorithms - index) for index, algorithm in enumerate(ranking)}
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")
