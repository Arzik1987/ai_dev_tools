"""Pairwise aggregators organized by the taxonomy families."""

from __future__ import annotations

from itertools import combinations
from statistics import fmean
from typing import Sequence

from rank_aggregation.base import AggregationResult, RankAggregator, Rankings
from rank_aggregation._methods.shared import (
    build_colley_system,
    build_massey_system,
    fit_bradley_terry,
    fit_polyrank_coefficients,
    fit_thurstone_mosteller,
    layered_defeat_ranking,
    least_squares_pairwise_scores,
    linear_ordering_exact,
    linear_ordering_heuristic,
    maximal_lottery_distribution,
    pairwise_margins,
    pairwise_supports,
    polyrank_observations,
    polyrank_transform,
    ranked_pairs_creates_cycle,
    ranked_pairs_linear_extension,
    reachable,
    recursive_pairwise_elimination,
    river_winner,
    schulze_strongest_paths,
    solve_linear_system,
    stationary_distribution,
    strongest_margin_paths,
)


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
        margins = pairwise_margins(rankings, algorithms)
        scores = {
            algorithm: float(sum(margins[(algorithm, opponent)] for opponent in algorithms if opponent != algorithm))
            for algorithm in algorithms
        }
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}
        ranking = sorted(algorithms, key=lambda algorithm: (-scores[algorithm], mean_ranks[algorithm], algorithm))
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

        duels.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        locked: dict[str, set[str]] = {algorithm: set() for algorithm in algorithms}

        for _, _, winner, loser in duels:
            if loser in locked[winner]:
                continue
            if ranked_pairs_creates_cycle(locked, winner, loser):
                continue
            locked[winner].add(loser)

        order = ranked_pairs_linear_extension(locked, algorithms)
        scores = {algorithm: float(index + 1) for index, algorithm in enumerate(order)}
        return AggregationResult(scores=scores, ranking=order)

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
        supports = pairwise_supports(rankings, algorithms)
        strongest_paths = schulze_strongest_paths(algorithms, supports)
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
        margins = pairwise_margins(rankings, algorithms)
        strongest_paths = strongest_margin_paths(algorithms, margins)
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

        ranking = layered_defeat_ranking(defeats, algorithms, mean_ranks)
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
        margins = pairwise_margins(rankings, algorithms)
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}

        remaining = algorithms[:]
        ranking: list[str] = []
        while remaining:
            winner = river_winner(remaining, margins, mean_ranks)
            ranking.append(winner)
            remaining.remove(winner)

        scores = {algorithm: float(len(algorithms) - index) for index, algorithm in enumerate(ranking)}
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
        margins = pairwise_margins(rankings, algorithms)
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}
        ranking = recursive_pairwise_elimination(
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
        margins = pairwise_margins(rankings, algorithms)
        mean_ranks = {algorithm: float(fmean(rankings[algorithm])) for algorithm in algorithms}
        ranking = recursive_pairwise_elimination(
            algorithms,
            margins,
            mean_ranks,
            use_stable_filter=False,
        )
        scores = {algorithm: float(len(algorithms) - index) for index, algorithm in enumerate(ranking)}
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


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
                    first_wins += 0.5
                    second_wins += 0.5
            wins[(first, second)] = first_wins
            wins[(second, first)] = second_wins

        scores = fit_bradley_terry(algorithms, wins, max_iter=self.max_iter, tol=self.tol)
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
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
        supports = pairwise_supports(rankings, algorithms)
        scores = fit_thurstone_mosteller(
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


class PolyRankAggregator(RankAggregator):
    """R-PolyRank: alternating fit of latent scores and an odd polynomial link."""

    def __init__(self, degree: int = 1, max_iter: int = 50, tol: float = 1e-8) -> None:
        if degree < 0:
            raise ValueError("degree must be >= 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if tol <= 0:
            raise ValueError("tol must be > 0")
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol

    @property
    def name(self) -> str:
        return "R-PolyRank"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        supports = pairwise_supports(rankings, algorithms)
        observations = polyrank_observations(algorithms, supports)
        if not observations:
            scores = {algorithm: 0.0 for algorithm in algorithms}
            return AggregationResult(scores=scores, ranking=algorithms)

        coefficients = [0.0 for _ in range(self.degree + 1)]
        coefficients[0] = 1.0
        previous_scores = {algorithm: 0.0 for algorithm in algorithms}

        for _ in range(self.max_iter):
            transformed = [polyrank_transform(observation["z"], coefficients) for observation in observations]
            score_values = least_squares_pairwise_scores(algorithms, observations, transformed)
            coefficients = fit_polyrank_coefficients(observations, score_values, algorithms, self.degree)
            coefficients[0] = max(coefficients[0], 1e-9)

            delta = max(abs(score_values[idx] - previous_scores[algorithm]) for idx, algorithm in enumerate(algorithms))
            previous_scores = {algorithm: score_values[idx] for idx, algorithm in enumerate(algorithms)}
            if delta < self.tol:
                break

        scores = previous_scores
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

        stationary = stationary_distribution(matrix, max_iter=self.max_iter, tol=self.tol)
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

        distribution = maximal_lottery_distribution(payoff, max_iter=self.max_iter, tol=self.tol)
        scores = {algorithm: distribution[idx] for idx, algorithm in enumerate(algorithms)}
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class MasseyRankingAggregator(RankAggregator):
    """R-Massey: least-squares pairwise ranking from signed rank margins."""

    @property
    def name(self) -> str:
        return "R-Massey"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        matrix, vector = build_massey_system(rankings, algorithms)
        solution = solve_linear_system(matrix, vector)
        scores = {algorithm: solution[idx] for idx, algorithm in enumerate(algorithms)}
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class ColleyRankingAggregator(RankAggregator):
    """R-Colley: regularized pairwise win/loss ranking from per-task duels."""

    @property
    def name(self) -> str:
        return "R-Colley"

    @property
    def higher_is_better(self) -> bool:
        return True

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        matrix, vector = build_colley_system(rankings, algorithms)
        solution = solve_linear_system(matrix, vector)
        scores = {algorithm: solution[idx] for idx, algorithm in enumerate(algorithms)}
        ranking = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        return AggregationResult(scores=scores, ranking=ranking)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")


class LinearOrderingProblemAggregator(RankAggregator):
    """R-LOP: maximize agreement with weighted pairwise preferences."""

    def __init__(self, exact_max_algorithms: int = 9) -> None:
        if exact_max_algorithms < 2:
            raise ValueError("exact_max_algorithms must be >= 2")
        self.exact_max_algorithms = exact_max_algorithms

    @property
    def name(self) -> str:
        return "R-LOP"

    def aggregate(self, rankings: Rankings) -> AggregationResult:
        self._validate(rankings)
        algorithms = sorted(rankings)
        supports = pairwise_supports(rankings, algorithms)

        if len(algorithms) <= self.exact_max_algorithms:
            order = linear_ordering_exact(algorithms, supports)
        else:
            order = linear_ordering_heuristic(rankings, algorithms, supports)

        scores = {algorithm: float(index + 1) for index, algorithm in enumerate(order)}
        return AggregationResult(scores=scores, ranking=order)

    def _score(self, ranks: Sequence[float]) -> float:
        raise NotImplementedError("aggregate() is overridden for this method")
