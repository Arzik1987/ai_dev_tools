"""MCDA and outranking aggregators."""

from __future__ import annotations

from math import sqrt
from statistics import fmean
from typing import Sequence

from rank_aggregation.base import AggregationResult, RankAggregator, Rankings
from rank_aggregation._methods.shared import (
    electre_partial_concordance,
    electre_partial_discordance,
    expanded_parameter,
    normalized_weights,
    quality_matrix,
    usual_preference_index,
)


class PROMETHEEIIAggregator(RankAggregator):
    """R-P2: PROMETHEE II ranking using the usual preference criterion."""

    def __init__(self, weights: Sequence[float] | None = None) -> None:
        self.weights = weights

    @property
    def name(self) -> str:
        return "R-P2"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        values = quality_matrix(rankings, algorithms)
        weights = normalized_weights(self.weights, len(values[0]))

        positive_flow = {algorithm: 0.0 for algorithm in algorithms}
        negative_flow = {algorithm: 0.0 for algorithm in algorithms}
        for first_idx, first in enumerate(algorithms):
            for second_idx, second in enumerate(algorithms):
                if first_idx == second_idx:
                    continue
                preference = 0.0
                for task_idx, weight in enumerate(weights):
                    if values[first_idx][task_idx] > values[second_idx][task_idx]:
                        preference += weight
                positive_flow[first] += preference
                negative_flow[first] += usual_preference_index(values[second_idx], values[first_idx], weights)

        divisor = max(1, len(algorithms) - 1)
        scores = {
            algorithm: (positive_flow[algorithm] - negative_flow[algorithm]) / divisor
            for algorithm in algorithms
        }
        outgoing = {algorithm: positive_flow[algorithm] / divisor for algorithm in algorithms}
        mean_quality = {algorithm: float(fmean(values[idx])) for idx, algorithm in enumerate(algorithms)}
        ranking = sorted(
            algorithms,
            key=lambda algorithm: (-scores[algorithm], -outgoing[algorithm], -mean_quality[algorithm], algorithm),
        )
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class ELECTREIIIAggregator(RankAggregator):
    """R-E3: ELECTRE III-style outranking aggregated into net credibility flows."""

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        q: float | Sequence[float] = 0.0,
        p: float | Sequence[float] = 0.0,
        v: float | Sequence[float] | None = None,
    ) -> None:
        self.weights = weights
        self.q = q
        self.p = p
        self.v = v

    @property
    def name(self) -> str:
        return "R-E3"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        values = quality_matrix(rankings, algorithms)
        task_count = len(values[0])
        weights = normalized_weights(self.weights, task_count)
        q_values = expanded_parameter(self.q, task_count, "q")
        p_values = expanded_parameter(self.p, task_count, "p")
        v_values = None if self.v is None else expanded_parameter(self.v, task_count, "v")

        for task_idx in range(task_count):
            if q_values[task_idx] < 0 or p_values[task_idx] < 0:
                raise ValueError("q and p must be non-negative")
            if q_values[task_idx] > p_values[task_idx]:
                raise ValueError("q must be <= p for every task")
            if v_values is not None:
                if v_values[task_idx] < 0:
                    raise ValueError("v must be non-negative")
                if p_values[task_idx] > v_values[task_idx]:
                    raise ValueError("p must be <= v for every task")

        credibility: dict[tuple[str, str], float] = {}
        for first_idx, first in enumerate(algorithms):
            for second_idx, second in enumerate(algorithms):
                if first_idx == second_idx:
                    continue
                partial_concordance: list[float] = []
                partial_discordance: list[float] = []
                for task_idx in range(task_count):
                    diff = values[first_idx][task_idx] - values[second_idx][task_idx]
                    partial_concordance.append(
                        electre_partial_concordance(diff, q_values[task_idx], p_values[task_idx])
                    )
                    if v_values is None:
                        partial_discordance.append(0.0)
                    else:
                        partial_discordance.append(
                            electre_partial_discordance(diff, p_values[task_idx], v_values[task_idx])
                        )

                concordance = sum(weight * value for weight, value in zip(weights, partial_concordance))
                score = concordance
                if concordance < 1.0:
                    for discordance in partial_discordance:
                        if discordance > concordance:
                            score *= (1.0 - discordance) / (1.0 - concordance)
                credibility[(first, second)] = score

        divisor = max(1, len(algorithms) - 1)
        scores = {
            algorithm: (
                sum(credibility[(algorithm, opponent)] for opponent in algorithms if opponent != algorithm)
                - sum(credibility[(opponent, algorithm)] for opponent in algorithms if opponent != algorithm)
            )
            / divisor
            for algorithm in algorithms
        }
        outgoing = {
            algorithm: sum(credibility[(algorithm, opponent)] for opponent in algorithms if opponent != algorithm)
            / divisor
            for algorithm in algorithms
        }
        mean_quality = {algorithm: float(fmean(values[idx])) for idx, algorithm in enumerate(algorithms)}
        ranking = sorted(
            algorithms,
            key=lambda algorithm: (-scores[algorithm], -outgoing[algorithm], -mean_quality[algorithm], algorithm),
        )
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class TOPSISAggregator(RankAggregator):
    """R-TOPSIS: TOPSIS closeness-coefficient ranking."""

    def __init__(self, weights: Sequence[float] | None = None) -> None:
        self.weights = weights

    @property
    def name(self) -> str:
        return "R-TOPSIS"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        values = quality_matrix(rankings, algorithms)
        task_count = len(values[0])
        weights = normalized_weights(self.weights, task_count)

        normalized = [[0.0 for _ in range(task_count)] for _ in algorithms]
        for task_idx in range(task_count):
            norm = sqrt(sum(row[task_idx] * row[task_idx] for row in values))
            if norm == 0:
                continue
            for algorithm_idx in range(len(algorithms)):
                normalized[algorithm_idx][task_idx] = values[algorithm_idx][task_idx] / norm

        weighted = [
            [normalized[algorithm_idx][task_idx] * weights[task_idx] for task_idx in range(task_count)]
            for algorithm_idx in range(len(algorithms))
        ]
        ideal_best = [max(row[task_idx] for row in weighted) for task_idx in range(task_count)]
        ideal_worst = [min(row[task_idx] for row in weighted) for task_idx in range(task_count)]

        scores: dict[str, float] = {}
        for algorithm_idx, algorithm in enumerate(algorithms):
            distance_best = sqrt(
                sum((weighted[algorithm_idx][task_idx] - ideal_best[task_idx]) ** 2 for task_idx in range(task_count))
            )
            distance_worst = sqrt(
                sum((weighted[algorithm_idx][task_idx] - ideal_worst[task_idx]) ** 2 for task_idx in range(task_count))
            )
            total_distance = distance_best + distance_worst
            scores[algorithm] = 0.0 if total_distance == 0 else distance_worst / total_distance

        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class VIKORAggregator(RankAggregator):
    """R-VIKOR: VIKOR compromise ranking."""

    def __init__(self, weights: Sequence[float] | None = None, v: float = 0.5) -> None:
        if v < 0 or v > 1:
            raise ValueError("v must be in [0, 1]")
        self.weights = weights
        self.v = v

    @property
    def name(self) -> str:
        return "R-VIKOR"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        values = quality_matrix(rankings, algorithms)
        task_count = len(values[0])
        weights = normalized_weights(self.weights, task_count)

        best = [max(row[task_idx] for row in values) for task_idx in range(task_count)]
        worst = [min(row[task_idx] for row in values) for task_idx in range(task_count)]
        s_scores: dict[str, float] = {}
        r_scores: dict[str, float] = {}
        for algorithm_idx, algorithm in enumerate(algorithms):
            contributions: list[float] = []
            for task_idx in range(task_count):
                span = best[task_idx] - worst[task_idx]
                if span == 0:
                    contributions.append(0.0)
                else:
                    contributions.append(weights[task_idx] * (best[task_idx] - values[algorithm_idx][task_idx]) / span)
            s_scores[algorithm] = float(sum(contributions))
            r_scores[algorithm] = float(max(contributions, default=0.0))

        s_best = min(s_scores.values())
        s_worst = max(s_scores.values())
        r_best = min(r_scores.values())
        r_worst = max(r_scores.values())
        scores: dict[str, float] = {}
        for algorithm in algorithms:
            s_term = 0.0 if s_worst == s_best else (s_scores[algorithm] - s_best) / (s_worst - s_best)
            r_term = 0.0 if r_worst == r_best else (r_scores[algorithm] - r_best) / (r_worst - r_best)
            scores[algorithm] = self.v * s_term + (1.0 - self.v) * r_term

        ranking = sorted(
            algorithms,
            key=lambda algorithm: (scores[algorithm], s_scores[algorithm], r_scores[algorithm], algorithm),
        )
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")
