"""Aggregation methods described in README."""

from __future__ import annotations

from itertools import combinations, permutations
from math import exp, log, sqrt
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


class BradleyTerryAggregator(RankAggregator):
    """R-BT: Bradley-Terry pairwise comparative rating."""

    def __init__(self, max_iter: int = 1000, tol: float = 1e-10) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if tol <= 0:
            raise ValueError("tol must be > 0")
        self.max_iter = max_iter
        self.tol = tol

    @property
    def name(self) -> str:
        return "R-BT"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        task_count = len(next(iter(rankings.values())))

        wins: dict[tuple[str, str], float] = {(first, second): 0.0 for first in algorithms for second in algorithms}
        for first, second in combinations(algorithms, 2):
            first_wins = 0.0
            second_wins = 0.0
            for task_idx in range(task_count):
                first_rank = rankings[first][task_idx]
                second_rank = rankings[second][task_idx]
                if first_rank < second_rank:
                    first_wins += 1.0
                elif second_rank < first_rank:
                    second_wins += 1.0
                else:
                    # Split ties equally to keep pairwise totals consistent.
                    first_wins += 0.5
                    second_wins += 0.5
            wins[(first, second)] = first_wins
            wins[(second, first)] = second_wins

        scores = _fit_bradley_terry(algorithms, wins, max_iter=self.max_iter, tol=self.tol)
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class SchulzeBeatpathAggregator(RankAggregator):
    """R-Sch: Schulze strongest-path aggregation from pairwise support."""

    @property
    def name(self) -> str:
        return "R-Sch"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        supports = _pairwise_supports(rankings, algorithms)
        strongest_paths = _schulze_strongest_paths(algorithms, supports)
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}

        scores = {algorithm: 0.0 for algorithm in algorithms}
        for first in algorithms:
            for second in algorithms:
                if first == second:
                    continue
                if strongest_paths[(first, second)] > strongest_paths[(second, first)]:
                    scores[first] += 1.0

        ranking = sorted(algorithms, key=lambda algorithm: (-scores[algorithm], mean_ranks[algorithm], algorithm))
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class MarginRowSumAggregator(RankAggregator):
    """R-MRS: sum of pairwise row margins against all opponents."""

    @property
    def name(self) -> str:
        return "R-MRS"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        margins = _pairwise_margins(rankings, algorithms)
        scores = {
            algorithm: float(sum(margins[(algorithm, opponent)] for opponent in algorithms if opponent != algorithm))
            for algorithm in algorithms
        }
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}
        ranking = sorted(algorithms, key=lambda algorithm: (-scores[algorithm], mean_ranks[algorithm], algorithm))
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class SplitCycleAggregator(RankAggregator):
    """R-SC: Split Cycle ranking from pairwise margin defeats."""

    @property
    def name(self) -> str:
        return "R-SC"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        margins = _pairwise_margins(rankings, algorithms)
        strongest_paths = _strongest_margin_paths(algorithms, margins)
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}

        defeats: dict[str, set[str]] = {algorithm: set() for algorithm in algorithms}
        scores = {algorithm: 0.0 for algorithm in algorithms}
        for winner in algorithms:
            for loser in algorithms:
                if winner == loser:
                    continue
                if margins[(winner, loser)] > strongest_paths[(loser, winner)]:
                    defeats[winner].add(loser)
                    scores[winner] += 1.0

        ranking = _layered_defeat_ranking(defeats, algorithms, mean_ranks)
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class RiverAggregator(RankAggregator):
    """R-River: recursive River winner elimination from pairwise margins."""

    @property
    def name(self) -> str:
        return "R-River"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        margins = _pairwise_margins(rankings, algorithms)
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}

        remaining = algorithms[:]
        ranking: list[str] = []
        while remaining:
            winner = _river_winner(remaining, margins, mean_ranks)
            ranking.append(winner)
            remaining.remove(winner)

        scores = {algorithm: float(len(algorithms) - index) for index, algorithm in enumerate(ranking)}
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class ThurstoneMostellerAggregator(RankAggregator):
    """R-TM: Thurstone-Mosteller paired-comparison model with probit link."""

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-10,
        ridge: float = 1e-6,
        initial_step: float = 1.0,
    ) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if tol <= 0:
            raise ValueError("tol must be > 0")
        if ridge < 0:
            raise ValueError("ridge must be >= 0")
        if initial_step <= 0:
            raise ValueError("initial_step must be > 0")
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.initial_step = initial_step

    @property
    def name(self) -> str:
        return "R-TM"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        supports = _pairwise_supports(rankings, algorithms)
        scores = _fit_thurstone_mosteller(
            algorithms,
            supports,
            max_iter=self.max_iter,
            tol=self.tol,
            ridge=self.ridge,
            initial_step=self.initial_step,
        )
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class StableVotingAggregator(RankAggregator):
    """R-SV: Stable Voting with Split-Cycle-style admissibility."""

    @property
    def name(self) -> str:
        return "R-SV"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        margins = _pairwise_margins(rankings, algorithms)
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}
        ranking = _recursive_pairwise_elimination(
            algorithms,
            margins,
            mean_ranks,
            use_stable_filter=True,
        )
        scores = {algorithm: float(len(algorithms) - index) for index, algorithm in enumerate(ranking)}
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class SimpleStableVotingAggregator(RankAggregator):
    """R-SSV: Simple Stable Voting recursive elimination."""

    @property
    def name(self) -> str:
        return "R-SSV"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        margins = _pairwise_margins(rankings, algorithms)
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}
        ranking = _recursive_pairwise_elimination(
            algorithms,
            margins,
            mean_ranks,
            use_stable_filter=False,
        )
        scores = {algorithm: float(len(algorithms) - index) for index, algorithm in enumerate(ranking)}
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


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

        scores = _fit_plackett_luce(algorithms, ordered_tasks, max_iter=self.max_iter, tol=self.tol)
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class CopelandPairwiseAggregator(RankAggregator):
    """R-Cop: Copeland pairwise majority method."""

    @property
    def name(self) -> str:
        return "R-Cop"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        task_count = len(next(iter(rankings.values())))
        scores = {algorithm: 0.0 for algorithm in algorithms}

        for first, second in combinations(algorithms, 2):
            first_wins = 0
            second_wins = 0
            for task_idx in range(task_count):
                first_rank = rankings[first][task_idx]
                second_rank = rankings[second][task_idx]
                if first_rank < second_rank:
                    first_wins += 1
                elif second_rank < first_rank:
                    second_wins += 1

            if first_wins > second_wins:
                scores[first] += 1.0
            elif second_wins > first_wins:
                scores[second] += 1.0
            else:
                scores[first] += 0.5
                scores[second] += 0.5

        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class MarkovChainAggregator(RankAggregator):
    """R-MC: Markov-chain rank aggregation from pairwise superiority counts."""

    def __init__(self, damping: float = 0.15, max_iter: int = 1000, tol: float = 1e-12) -> None:
        if damping < 0 or damping >= 1:
            raise ValueError("damping must be in [0, 1)")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if tol <= 0:
            raise ValueError("tol must be > 0")
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    @property
    def name(self) -> str:
        return "R-MC"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        task_count = len(next(iter(rankings.values())))
        n_algorithms = len(algorithms)

        matrix: list[list[float]] = [[0.0 for _ in range(n_algorithms)] for _ in range(n_algorithms)]
        for source_idx, source in enumerate(algorithms):
            superior_counts: list[int] = [0 for _ in range(n_algorithms)]
            denom = 0
            for target_idx, target in enumerate(algorithms):
                if source_idx == target_idx:
                    continue
                count = 0
                for task_idx in range(task_count):
                    if rankings[target][task_idx] < rankings[source][task_idx]:
                        count += 1
                superior_counts[target_idx] = count
                denom += count

            if denom == 0:
                for target_idx in range(n_algorithms):
                    matrix[source_idx][target_idx] = 1.0 / n_algorithms
            else:
                for target_idx in range(n_algorithms):
                    matrix[source_idx][target_idx] = superior_counts[target_idx] / denom

            if self.damping > 0:
                teleport = self.damping / n_algorithms
                for target_idx in range(n_algorithms):
                    matrix[source_idx][target_idx] = (1.0 - self.damping) * matrix[source_idx][target_idx] + teleport

        stationary = _stationary_distribution(matrix, max_iter=self.max_iter, tol=self.tol)
        scores = {algorithm: stationary[idx] for idx, algorithm in enumerate(algorithms)}
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class MaximalLotteryAggregator(RankAggregator):
    """R-ML: maximal-lottery style mixed strategy from pairwise margins."""

    def __init__(self, max_iter: int = 10000, tol: float = 1e-6) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if tol <= 0:
            raise ValueError("tol must be > 0")
        self.max_iter = max_iter
        self.tol = tol

    @property
    def name(self) -> str:
        return "R-ML"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        task_count = len(next(iter(rankings.values())))
        n_algorithms = len(algorithms)

        payoff = [[0.0 for _ in range(n_algorithms)] for _ in range(n_algorithms)]
        for left in range(n_algorithms):
            for right in range(left + 1, n_algorithms):
                left_wins = 0
                right_wins = 0
                for task_idx in range(task_count):
                    left_rank = rankings[algorithms[left]][task_idx]
                    right_rank = rankings[algorithms[right]][task_idx]
                    if left_rank < right_rank:
                        left_wins += 1
                    elif right_rank < left_rank:
                        right_wins += 1
                margin = float(left_wins - right_wins)
                payoff[left][right] = margin
                payoff[right][left] = -margin

        distribution = _maximal_lottery_distribution(payoff, max_iter=self.max_iter, tol=self.tol)
        scores = {algorithm: distribution[idx] for idx, algorithm in enumerate(algorithms)}
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class MinimaxCondorcetAggregator(RankAggregator):
    """R-Minimax: minimize worst pairwise loss count across opponents."""

    @property
    def name(self) -> str:
        return "R-Minimax"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        task_count = len(next(iter(rankings.values())))
        scores = {algorithm: 0.0 for algorithm in algorithms}

        for algorithm in algorithms:
            worst_loss = 0
            for opponent in algorithms:
                if algorithm == opponent:
                    continue
                losses = 0
                for task_idx in range(task_count):
                    if rankings[opponent][task_idx] < rankings[algorithm][task_idx]:
                        losses += 1
                worst_loss = max(worst_loss, losses)
            scores[algorithm] = float(worst_loss)

        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class RankedPairsTidemanAggregator(RankAggregator):
    """R-RP: Ranked Pairs (Tideman) consensus from pairwise victories."""

    @property
    def name(self) -> str:
        return "R-RP"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        task_count = len(next(iter(rankings.values())))

        duels: list[tuple[int, int, str, str]] = []
        for first, second in combinations(algorithms, 2):
            first_wins = 0
            second_wins = 0
            for task_idx in range(task_count):
                first_rank = rankings[first][task_idx]
                second_rank = rankings[second][task_idx]
                if first_rank < second_rank:
                    first_wins += 1
                elif second_rank < first_rank:
                    second_wins += 1

            if first_wins > second_wins:
                duels.append((first_wins - second_wins, first_wins, first, second))
            elif second_wins > first_wins:
                duels.append((second_wins - first_wins, second_wins, second, first))

        # Stronger victories are locked first; deterministic tie-break by winner/loser id.
        duels.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        locked: dict[str, set[str]] = {algorithm: set() for algorithm in algorithms}

        for _, _, winner, loser in duels:
            if loser in locked[winner]:
                continue
            if _ranked_pairs_creates_cycle(locked, winner, loser):
                continue
            locked[winner].add(loser)

        order = _ranked_pairs_linear_extension(locked, algorithms)
        scores = {algorithm: float(index + 1) for index, algorithm in enumerate(order)}
        return AggregationResult(scores=scores, ranking=order)

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
        scores = _dm_auc_scores(
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
            auc_scores = _dm_auc_scores(
                rankings,
                remaining,
                tau_max=self.tau_max,
                normalize_auc=self.normalize_auc,
            )
            winner = min(remaining, key=lambda algorithm: (-auc_scores[algorithm], algorithm))
            ranking.append(winner)
            remaining.remove(winner)

        # Round-based positional points preserve the iterative ordering.
        scores = {
            algorithm: float(total_algorithms - index)
            for index, algorithm in enumerate(ranking)
        }
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


def _fit_bradley_terry(
    algorithms: list[str],
    wins: dict[tuple[str, str], float],
    max_iter: int,
    tol: float,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths with MM updates."""
    k = len(algorithms)
    strengths = {algorithm: 1.0 / k for algorithm in algorithms}
    totals = {
        algorithm: sum(wins[(algorithm, opponent)] for opponent in algorithms if opponent != algorithm)
        for algorithm in algorithms
    }

    for _ in range(max_iter):
        updated: dict[str, float] = {}
        for algorithm in algorithms:
            denominator = 0.0
            for opponent in algorithms:
                if algorithm == opponent:
                    continue
                n_ij = wins[(algorithm, opponent)] + wins[(opponent, algorithm)]
                denominator += n_ij / (strengths[algorithm] + strengths[opponent])
            if denominator == 0:
                updated[algorithm] = strengths[algorithm]
            else:
                updated[algorithm] = totals[algorithm] / denominator

        total_strength = sum(updated.values())
        if total_strength <= 0:
            return {algorithm: 1.0 / k for algorithm in algorithms}
        for algorithm in algorithms:
            updated[algorithm] /= total_strength

        delta = max(abs(updated[algorithm] - strengths[algorithm]) for algorithm in algorithms)
        strengths = updated
        if delta < tol:
            break
    return strengths


def _fit_plackett_luce(
    algorithms: list[str],
    ordered_tasks: list[list[str]],
    max_iter: int,
    tol: float,
) -> dict[str, float]:
    """Fit Plackett-Luce worth parameters with MM updates."""
    n_algorithms = len(algorithms)
    worth = {algorithm: 1.0 / n_algorithms for algorithm in algorithms}
    wins = {algorithm: float(len(ordered_tasks)) for algorithm in algorithms}

    for _ in range(max_iter):
        denominators = {algorithm: 0.0 for algorithm in algorithms}
        for order in ordered_tasks:
            for position in range(len(order) - 1):
                risk_set = order[position:]
                normalizer = sum(worth[item] for item in risk_set)
                if normalizer <= 0:
                    continue
                contribution = 1.0 / normalizer
                for item in risk_set:
                    denominators[item] += contribution

        updated: dict[str, float] = {}
        for algorithm in algorithms:
            denominator = denominators[algorithm]
            if denominator <= 0:
                updated[algorithm] = worth[algorithm]
            else:
                updated[algorithm] = wins[algorithm] / denominator

        total = sum(updated.values())
        if total <= 0:
            return {algorithm: 1.0 / n_algorithms for algorithm in algorithms}
        for algorithm in algorithms:
            updated[algorithm] /= total

        delta = max(abs(updated[algorithm] - worth[algorithm]) for algorithm in algorithms)
        worth = updated
        if delta < tol:
            break
    return worth


def _fit_thurstone_mosteller(
    algorithms: list[str],
    wins: dict[tuple[str, str], float],
    max_iter: int,
    tol: float,
    ridge: float,
    initial_step: float,
) -> dict[str, float]:
    """Fit zero-centered Thurstone-Mosteller strengths via projected gradient ascent."""
    strengths = {algorithm: 0.0 for algorithm in algorithms}

    for _ in range(max_iter):
        baseline = _thurstone_mosteller_objective(algorithms, wins, strengths, ridge)
        gradient = {algorithm: 0.0 for algorithm in algorithms}

        for first, second in combinations(algorithms, 2):
            diff = strengths[first] - strengths[second]
            prob = min(max(NormalDist().cdf(diff), 1e-12), 1.0 - 1e-12)
            density = _standard_normal_pdf(diff)
            common = density * (wins[(first, second)] / prob - wins[(second, first)] / (1.0 - prob))
            gradient[first] += common
            gradient[second] -= common

        if ridge > 0:
            for algorithm in algorithms:
                gradient[algorithm] -= 2.0 * ridge * strengths[algorithm]

        mean_gradient = fmean(gradient.values())
        for algorithm in algorithms:
            gradient[algorithm] -= mean_gradient

        step = initial_step
        updated = strengths
        improved = False
        while step > 1e-12:
            candidate = {algorithm: strengths[algorithm] + step * gradient[algorithm] for algorithm in algorithms}
            mean_candidate = fmean(candidate.values())
            for algorithm in algorithms:
                candidate[algorithm] -= mean_candidate

            objective = _thurstone_mosteller_objective(algorithms, wins, candidate, ridge)
            if objective >= baseline:
                updated = candidate
                improved = True
                break
            step *= 0.5

        delta = max(abs(updated[algorithm] - strengths[algorithm]) for algorithm in algorithms)
        strengths = updated
        if not improved or delta < tol:
            break

    return strengths


def _stationary_distribution(matrix: list[list[float]], max_iter: int, tol: float) -> list[float]:
    """Compute stationary distribution for a row-stochastic matrix via power iteration."""
    n = len(matrix)
    if n == 0:
        return []
    state = [1.0 / n for _ in range(n)]
    for _ in range(max_iter):
        updated = [0.0 for _ in range(n)]
        for source_idx in range(n):
            source_prob = state[source_idx]
            row = matrix[source_idx]
            for target_idx in range(n):
                updated[target_idx] += source_prob * row[target_idx]
        total = sum(updated)
        if total > 0:
            updated = [value / total for value in updated]
        delta = max(abs(updated[idx] - state[idx]) for idx in range(n))
        state = updated
        if delta < tol:
            break
    return state


def _maximal_lottery_distribution(payoff: list[list[float]], max_iter: int, tol: float) -> list[float]:
    """Approximate a maximal lottery via deterministic fictitious play."""
    n = len(payoff)
    if n == 0:
        return []
    row_counts = [1.0 for _ in range(n)]
    col_counts = [1.0 for _ in range(n)]

    for _ in range(max_iter):
        row_total = sum(row_counts)
        col_total = sum(col_counts)
        row_strategy = [value / row_total for value in row_counts]
        col_strategy = [value / col_total for value in col_counts]

        row_payoffs = [sum(payoff[i][j] * col_strategy[j] for j in range(n)) for i in range(n)]
        col_values = [sum(row_strategy[i] * payoff[i][j] for i in range(n)) for j in range(n)]
        exploitability = max(row_payoffs) - min(col_values)
        if exploitability < tol:
            break

        best_row = max(range(n), key=lambda i: row_payoffs[i])
        best_col = min(range(n), key=lambda j: col_values[j])
        row_counts[best_row] += 1.0
        col_counts[best_col] += 1.0

    total = sum(row_counts)
    if total <= 0:
        return [1.0 / n for _ in range(n)]
    return [value / total for value in row_counts]


def _thurstone_mosteller_objective(
    algorithms: Sequence[str],
    wins: dict[tuple[str, str], float],
    strengths: dict[str, float],
    ridge: float,
) -> float:
    """Penalized log-likelihood under the probit paired-comparison model."""
    objective = 0.0
    for first, second in combinations(algorithms, 2):
        diff = strengths[first] - strengths[second]
        prob = min(max(NormalDist().cdf(diff), 1e-12), 1.0 - 1e-12)
        objective += wins[(first, second)] * log(prob)
        objective += wins[(second, first)] * log(1.0 - prob)

    if ridge > 0:
        objective -= ridge * sum(value * value for value in strengths.values())
    return objective


def _standard_normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * 3.141592653589793)


def _dm_auc_scores(
    rankings: Rankings,
    algorithms: Sequence[str],
    tau_max: float | None,
    normalize_auc: bool,
) -> dict[str, float]:
    """Compute per-algorithm DM performance-profile AUC for a candidate pool."""
    task_count = len(next(iter(rankings.values())))
    ratios_by_algorithm: dict[str, list[float]] = {algorithm: [] for algorithm in algorithms}

    for task_idx in range(task_count):
        best_value = min(rankings[algorithm][task_idx] for algorithm in algorithms)
        for algorithm in algorithms:
            ratios_by_algorithm[algorithm].append(rankings[algorithm][task_idx] / best_value)

    effective_tau_max = tau_max
    if effective_tau_max is None:
        effective_tau_max = max(max(ratios) for ratios in ratios_by_algorithm.values())
    effective_tau_max = max(1.0, effective_tau_max)

    auc_scores: dict[str, float] = {}
    interval = effective_tau_max - 1.0
    for algorithm, ratios in ratios_by_algorithm.items():
        auc = 0.0
        sorted_ratios = sorted(ratios)
        cursor = 1.0
        seen = 0
        for ratio in sorted_ratios:
            clamped_ratio = min(max(1.0, ratio), effective_tau_max)
            if clamped_ratio > cursor:
                auc += (clamped_ratio - cursor) * (seen / task_count)
                cursor = clamped_ratio
            seen += 1
            if cursor >= effective_tau_max:
                break
        if cursor < effective_tau_max:
            auc += (effective_tau_max - cursor) * (seen / task_count)

        if normalize_auc and interval > 0:
            auc /= interval
        auc_scores[algorithm] = auc

    return auc_scores


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


def _pairwise_supports(rankings: Rankings, algorithms: Sequence[str]) -> dict[tuple[str, str], float]:
    """Return pairwise support counts, splitting tied tasks evenly."""
    supports: dict[tuple[str, str], float] = {(first, second): 0.0 for first in algorithms for second in algorithms}
    task_count = len(next(iter(rankings.values())))
    for first, second in combinations(algorithms, 2):
        first_support = 0.0
        second_support = 0.0
        for task_idx in range(task_count):
            first_rank = rankings[first][task_idx]
            second_rank = rankings[second][task_idx]
            if first_rank < second_rank:
                first_support += 1.0
            elif second_rank < first_rank:
                second_support += 1.0
            else:
                first_support += 0.5
                second_support += 0.5
        supports[(first, second)] = first_support
        supports[(second, first)] = second_support
    return supports


def _pairwise_margins(rankings: Rankings, algorithms: Sequence[str]) -> dict[tuple[str, str], float]:
    """Return antisymmetric pairwise win-count margins."""
    supports = _pairwise_supports(rankings, algorithms)
    margins: dict[tuple[str, str], float] = {}
    for first in algorithms:
        for second in algorithms:
            if first == second:
                margins[(first, second)] = 0.0
            else:
                margins[(first, second)] = supports[(first, second)] - supports[(second, first)]
    return margins


def _schulze_strongest_paths(
    algorithms: Sequence[str],
    supports: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """Compute strongest-path strengths using the Schulze winning-votes variant."""
    strongest_paths: dict[tuple[str, str], float] = {(first, second): 0.0 for first in algorithms for second in algorithms}

    for first in algorithms:
        for second in algorithms:
            if first == second:
                continue
            if supports[(first, second)] > supports[(second, first)]:
                strongest_paths[(first, second)] = supports[(first, second)]

    for pivot in algorithms:
        for source in algorithms:
            if source == pivot:
                continue
            for target in algorithms:
                if target == source or target == pivot:
                    continue
                strongest_paths[(source, target)] = max(
                    strongest_paths[(source, target)],
                    min(strongest_paths[(source, pivot)], strongest_paths[(pivot, target)]),
                )

    return strongest_paths


def _strongest_margin_paths(
    algorithms: Sequence[str],
    margins: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """Compute strongest path strengths using positive pairwise margins as edge weights."""
    strongest_paths: dict[tuple[str, str], float] = {(first, second): 0.0 for first in algorithms for second in algorithms}

    for first in algorithms:
        for second in algorithms:
            if first == second:
                continue
            strongest_paths[(first, second)] = max(0.0, margins[(first, second)])

    for pivot in algorithms:
        for source in algorithms:
            if source == pivot:
                continue
            for target in algorithms:
                if target == source or target == pivot:
                    continue
                strongest_paths[(source, target)] = max(
                    strongest_paths[(source, target)],
                    min(strongest_paths[(source, pivot)], strongest_paths[(pivot, target)]),
                )

    return strongest_paths


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


def _layered_defeat_ranking(
    defeats: dict[str, set[str]],
    algorithms: Sequence[str],
    mean_ranks: dict[str, float],
) -> list[str]:
    """Produce a deterministic order by repeatedly removing undefeated layers."""
    remaining = set(algorithms)
    ranking: list[str] = []
    while remaining:
        layer = [
            algorithm
            for algorithm in remaining
            if not any(algorithm in defeats[opponent] for opponent in remaining if opponent != algorithm)
        ]
        if not layer:
            layer = list(remaining)
        layer.sort(key=lambda algorithm: (mean_ranks[algorithm], algorithm))
        ranking.extend(layer)
        for algorithm in layer:
            remaining.remove(algorithm)
    return ranking


def _reachable(graph: dict[str, set[str]], source: str, target: str) -> bool:
    stack = [source]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph[current])
    return False


def _reachability_size(graph: dict[str, set[str]], source: str) -> int:
    stack = [source]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph[current])
    return len(seen)


def _river_winner(
    algorithms: Sequence[str],
    margins: dict[tuple[str, str], float],
    mean_ranks: dict[str, float],
) -> str:
    """Select the River winner for the current candidate subset."""
    graph: dict[str, set[str]] = {algorithm: set() for algorithm in algorithms}
    indegree = {algorithm: 0 for algorithm in algorithms}
    ordered_edges = sorted(
        [
            (margins[(winner, loser)], winner, loser)
            for winner in algorithms
            for loser in algorithms
            if winner != loser and margins[(winner, loser)] > 0
        ],
        key=lambda item: (-item[0], item[1], item[2]),
    )

    for _, winner, loser in ordered_edges:
        if indegree[winner] != 0:
            continue
        if indegree[loser] != 0:
            continue
        if _reachable(graph, loser, winner):
            continue
        graph[winner].add(loser)
        indegree[loser] = 1

    roots = [algorithm for algorithm in algorithms if indegree[algorithm] == 0]
    return min(
        roots,
        key=lambda algorithm: (-_reachability_size(graph, algorithm), mean_ranks[algorithm], algorithm),
    )


def _recursive_pairwise_elimination(
    algorithms: list[str],
    margins: dict[tuple[str, str], float],
    mean_ranks: dict[str, float],
    use_stable_filter: bool,
) -> list[str]:
    """Repeatedly remove memoized winner sets to produce a total order."""
    index = {algorithm: bit for bit, algorithm in enumerate(algorithms)}
    memo: dict[int, frozenset[str]] = {}

    def winners(mask: int) -> frozenset[str]:
        if mask in memo:
            return memo[mask]

        present = [algorithm for algorithm in algorithms if mask & (1 << index[algorithm])]
        if len(present) == 1:
            result = frozenset(present)
            memo[mask] = result
            return result

        condorcet = [
            algorithm
            for algorithm in present
            if all(
                algorithm == opponent or margins[(algorithm, opponent)] > 0
                for opponent in present
            )
        ]
        if condorcet:
            result = frozenset(condorcet)
            memo[mask] = result
            return result

        strongest_paths = _strongest_margin_paths(present, margins)
        pairs = sorted(
            [
                (margins[(winner, loser)], winner, loser)
                for winner in present
                for loser in present
                if winner != loser
            ],
            key=lambda item: (-item[0], item[1], item[2]),
        )

        best_margin: float | None = None
        result_set: set[str] = set()
        for margin, winner, loser in pairs:
            reduced_mask = mask & ~(1 << index[loser])
            if winner not in winners(reduced_mask):
                continue
            if use_stable_filter and margins[(loser, winner)] > 0:
                if strongest_paths[(winner, loser)] < margins[(loser, winner)]:
                    continue
            if best_margin is None:
                best_margin = margin
            if margin != best_margin:
                break
            result_set.add(winner)

        if not result_set:
            result_set.add(min(present, key=lambda algorithm: (mean_ranks[algorithm], algorithm)))

        result = frozenset(result_set)
        memo[mask] = result
        return result

    remaining = set(algorithms)
    ranking: list[str] = []
    while remaining:
        mask = 0
        for algorithm in remaining:
            mask |= 1 << index[algorithm]
        winner_set = list(winners(mask))
        winner_set.sort(key=lambda algorithm: (mean_ranks[algorithm], algorithm))
        chosen = winner_set[0]
        ranking.append(chosen)
        remaining.remove(chosen)
    return ranking


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


def _ranked_pairs_creates_cycle(
    locked: dict[str, set[str]],
    winner: str,
    loser: str,
) -> bool:
    """Return whether adding winner -> loser would introduce a cycle."""
    stack = [loser]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == winner:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(locked[current])
    return False


def _ranked_pairs_linear_extension(locked: dict[str, set[str]], algorithms: list[str]) -> list[str]:
    """Produce a deterministic total order from the locked DAG."""
    indegree = {algorithm: 0 for algorithm in algorithms}
    for winner in algorithms:
        for loser in locked[winner]:
            indegree[loser] += 1

    remaining = set(algorithms)
    order: list[str] = []
    while remaining:
        sources = [node for node in remaining if indegree[node] == 0]
        if not sources:
            # Should not happen with cycle checks, but keep deterministic fallback.
            sources = sorted(remaining)
        chosen = min(sources, key=lambda node: (-len(locked[node]), node))
        order.append(chosen)
        remaining.remove(chosen)
        for successor in locked[chosen]:
            indegree[successor] -= 1
    return order
