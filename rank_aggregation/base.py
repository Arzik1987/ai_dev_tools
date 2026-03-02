"""Shared structure for aggregation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence


Rankings = Mapping[str, Sequence[float]]


@dataclass(frozen=True)
class AggregationResult:
    """Aggregated scores and a deterministic ranking order."""

    scores: dict[str, float]
    ranking: list[str]


class RankAggregator(ABC):
    """Base class for aggregation strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def _score(self, ranks: Sequence[float]) -> float:
        """Compute the aggregate score for one algorithm."""

    @property
    def higher_is_better(self) -> bool:
        """Whether larger aggregate scores are preferred."""
        return False

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)

        scores = {algorithm: self._score(ranks) for algorithm, ranks in rankings.items()}
        if self.higher_is_better:
            ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        else:
            ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    @staticmethod
    def _validate(rankings: Rankings) -> None:
        if not rankings:
            raise ValueError("rankings must not be empty")

        lengths = {len(ranks) for ranks in rankings.values()}
        if lengths != {next(iter(lengths))}:
            raise ValueError("all algorithms must have the same number of task ranks")

        task_count = next(iter(lengths))
        if task_count == 0:
            raise ValueError("rank lists must not be empty")
