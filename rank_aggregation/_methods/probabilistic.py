"""Probabilistic rank-model aggregators."""

from __future__ import annotations

from typing import Sequence

from rank_aggregation.base import AggregationResult, RankAggregator, Rankings
from rank_aggregation._methods.shared import fit_plackett_luce


class PlackettLuceAggregator(RankAggregator):
    """R-PL: Plackett-Luce worth estimation from full task rankings."""

    def __init__(self, max_iter: int = 1000, tol: float = 1e-10) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if tol <= 0:
            raise ValueError("tol must be > 0")
        self.max_iter = max_iter
        self.tol = tol

    @property
    def name(self) -> str:
        return "R-PL"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        task_count = len(next(iter(rankings.values())))

        ordered_tasks: list[list[str]] = []
        for task_idx in range(task_count):
            ordered = sorted(algorithms, key=lambda algorithm: (rankings[algorithm][task_idx], algorithm))
            ordered_tasks.append(ordered)

        scores = fit_plackett_luce(algorithms, ordered_tasks, max_iter=self.max_iter, tol=self.tol)
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")
