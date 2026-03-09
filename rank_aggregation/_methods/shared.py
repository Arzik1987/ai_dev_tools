"""Shared helper functions for rank aggregation implementations."""

from __future__ import annotations

from itertools import combinations, permutations
from math import exp, log, sqrt
from statistics import NormalDist, fmean
from typing import Sequence

from rank_aggregation.base import Rankings

try:
    from scipy.stats import chi2, studentized_range
except Exception:  # pragma: no cover - optional dependency
    chi2 = None
    studentized_range = None


def quality_matrix(rankings: Rankings, algorithms: Sequence[str]) -> list[list[float]]:
    return [[float(value) for value in rankings[algorithm]] for algorithm in algorithms]


def normalized_weights(weights: Sequence[float] | None, size: int) -> list[float]:
    if size <= 0:
        return []
    if weights is None:
        return [1.0 / size for _ in range(size)]
    if len(weights) != size:
        raise ValueError("weights must match the number of tasks")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return [float(weight) / total for weight in weights]


def expanded_parameter(value: float | Sequence[float], size: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value) for _ in range(size)]
    expanded = [float(item) for item in value]
    if len(expanded) != size:
        raise ValueError(f"{name} must match the number of tasks")
    return expanded


def usual_preference_index(first: Sequence[float], second: Sequence[float], weights: Sequence[float]) -> float:
    return float(
        sum(weight for first_value, second_value, weight in zip(first, second, weights) if first_value > second_value)
    )


def electre_partial_concordance(diff: float, q_value: float, p_value: float) -> float:
    if diff >= -q_value:
        return 1.0
    if diff <= -p_value:
        return 0.0
    if p_value == q_value:
        return 0.0
    return (diff + p_value) / (p_value - q_value)


def electre_partial_discordance(diff: float, p_value: float, v_value: float) -> float:
    if diff >= -p_value:
        return 0.0
    if diff <= -v_value:
        return 1.0
    if v_value == p_value:
        return 1.0
    return (-diff - p_value) / (v_value - p_value)


def fit_bradley_terry(
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


def fit_plackett_luce(
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


def fit_thurstone_mosteller(
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
        baseline = thurstone_mosteller_objective(algorithms, wins, strengths, ridge)
        gradient = {algorithm: 0.0 for algorithm in algorithms}

        for first, second in combinations(algorithms, 2):
            diff = strengths[first] - strengths[second]
            prob = min(max(NormalDist().cdf(diff), 1e-12), 1.0 - 1e-12)
            density = standard_normal_pdf(diff)
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

            objective = thurstone_mosteller_objective(algorithms, wins, candidate, ridge)
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


def stationary_distribution(matrix: list[list[float]], max_iter: int, tol: float) -> list[float]:
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


def maximal_lottery_distribution(payoff: list[list[float]], max_iter: int, tol: float) -> list[float]:
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


def thurstone_mosteller_objective(
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


def standard_normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * 3.141592653589793)


def dm_auc_scores(
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


def chi_square_survival(statistic: float, df: int) -> float:
    """Return P(X >= statistic) for chi-square(df)."""
    if statistic <= 0:
        return 1.0
    if chi2 is not None:  # pragma: no branch
        return float(chi2.sf(statistic, df))

    if df <= 0:
        return 1.0
    transformed = (statistic / df) ** (1.0 / 3.0)
    mean = 1.0 - (2.0 / (9.0 * df))
    std = sqrt(2.0 / (9.0 * df))
    z = (transformed - mean) / std
    return max(0.0, min(1.0, 1.0 - NormalDist().cdf(z)))


def studentized_critical_value(alpha: float, k: int) -> float:
    """Critical value for Nemenyi comparisons."""
    if studentized_range is not None:  # pragma: no branch
        return float(studentized_range.isf(alpha, k, float("inf")))

    comparisons = max(1, (k * (k - 1)) // 2)
    per_comparison_alpha = alpha / comparisons
    z_value = NormalDist().inv_cdf(1 - per_comparison_alpha / 2)
    return sqrt(2.0) * z_value


def pairwise_supports(rankings: Rankings, algorithms: Sequence[str]) -> dict[tuple[str, str], float]:
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


def pairwise_margins(rankings: Rankings, algorithms: Sequence[str]) -> dict[tuple[str, str], float]:
    """Return antisymmetric pairwise win-count margins."""
    supports = pairwise_supports(rankings, algorithms)
    margins: dict[tuple[str, str], float] = {}
    for first in algorithms:
        for second in algorithms:
            if first == second:
                margins[(first, second)] = 0.0
            else:
                margins[(first, second)] = supports[(first, second)] - supports[(second, first)]
    return margins


def schulze_strongest_paths(
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


def strongest_margin_paths(
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


def pairwise_preferences(rankings: Rankings, algorithms: list[str]) -> dict[tuple[str, str], int]:
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


def layered_defeat_ranking(
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


def reachable(graph: dict[str, set[str]], source: str, target: str) -> bool:
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


def reachability_size(graph: dict[str, set[str]], source: str) -> int:
    stack = [source]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph[current])
    return len(seen)


def river_winner(
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
        if reachable(graph, loser, winner):
            continue
        graph[winner].add(loser)
        indegree[loser] = 1

    roots = [algorithm for algorithm in algorithms if indegree[algorithm] == 0]
    return min(
        roots,
        key=lambda algorithm: (-reachability_size(graph, algorithm), mean_ranks[algorithm], algorithm),
    )


def recursive_pairwise_elimination(
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
            if all(algorithm == opponent or margins[(algorithm, opponent)] > 0 for opponent in present)
        ]
        if condorcet:
            result = frozenset(condorcet)
            memo[mask] = result
            return result

        path_strengths = strongest_margin_paths(present, margins)
        pairs = sorted(
            [(margins[(winner, loser)], winner, loser) for winner in present for loser in present if winner != loser],
            key=lambda item: (-item[0], item[1], item[2]),
        )

        best_margin: float | None = None
        result_set: set[str] = set()
        for margin, winner, loser in pairs:
            reduced_mask = mask & ~(1 << index[loser])
            if winner not in winners(reduced_mask):
                continue
            if use_stable_filter and margins[(loser, winner)] > 0:
                if path_strengths[(winner, loser)] < margins[(loser, winner)]:
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


def kemeny_disagreement(order: Sequence[str], preferences: dict[tuple[str, str], int]) -> int:
    disagreement = 0
    for index, first in enumerate(order):
        for second in order[index + 1 :]:
            disagreement += preferences.get((second, first), 0)
    return disagreement


def kemeny_exact(algorithms: list[str], preferences: dict[tuple[str, str], int]) -> list[str]:
    best_order: tuple[str, ...] | None = None
    best_score: int | None = None

    for candidate in permutations(algorithms):
        score = kemeny_disagreement(candidate, preferences)
        if best_score is None or score < best_score or (score == best_score and candidate < best_order):
            best_order = candidate
            best_score = score

    return list(best_order) if best_order is not None else algorithms


def kemeny_heuristic(
    rankings: Rankings,
    algorithms: list[str],
    preferences: dict[tuple[str, str], int],
) -> list[str]:
    order = sorted(algorithms, key=lambda algorithm: (fmean(rankings[algorithm]), algorithm))
    best_score = kemeny_disagreement(order, preferences)

    improved = True
    while improved:
        improved = False
        for left in range(len(order) - 1):
            for right in range(left + 1, len(order)):
                candidate = order[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_score = kemeny_disagreement(candidate, preferences)
                if candidate_score < best_score:
                    order = candidate
                    best_score = candidate_score
                    improved = True
    return order


def ranked_pairs_creates_cycle(
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


def ranked_pairs_linear_extension(locked: dict[str, set[str]], algorithms: list[str]) -> list[str]:
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
            sources = sorted(remaining)
        chosen = min(sources, key=lambda node: (-len(locked[node]), node))
        order.append(chosen)
        remaining.remove(chosen)
        for successor in locked[chosen]:
            indegree[successor] -= 1
    return order


def build_massey_system(rankings: Rankings, algorithms: Sequence[str]) -> tuple[list[list[float]], list[float]]:
    size = len(algorithms)
    index = {algorithm: idx for idx, algorithm in enumerate(algorithms)}
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    vector = [0.0 for _ in range(size)]
    task_count = len(next(iter(rankings.values())))

    for first, second in combinations(algorithms, 2):
        first_idx = index[first]
        second_idx = index[second]
        encounters = 0
        margin_sum = 0.0
        for task_idx in range(task_count):
            first_rank = rankings[first][task_idx]
            second_rank = rankings[second][task_idx]
            encounters += 1
            margin_sum += second_rank - first_rank
        matrix[first_idx][first_idx] += encounters
        matrix[second_idx][second_idx] += encounters
        matrix[first_idx][second_idx] -= encounters
        matrix[second_idx][first_idx] -= encounters
        vector[first_idx] += margin_sum
        vector[second_idx] -= margin_sum

    matrix[-1] = [1.0 for _ in range(size)]
    vector[-1] = 0.0
    return matrix, vector


def build_colley_system(rankings: Rankings, algorithms: Sequence[str]) -> tuple[list[list[float]], list[float]]:
    size = len(algorithms)
    index = {algorithm: idx for idx, algorithm in enumerate(algorithms)}
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    wins = [0.0 for _ in range(size)]
    losses = [0.0 for _ in range(size)]
    comparisons = [[0.0 for _ in range(size)] for _ in range(size)]
    task_count = len(next(iter(rankings.values())))

    for first, second in combinations(algorithms, 2):
        first_idx = index[first]
        second_idx = index[second]
        for task_idx in range(task_count):
            comparisons[first_idx][second_idx] += 1.0
            comparisons[second_idx][first_idx] += 1.0
            first_rank = rankings[first][task_idx]
            second_rank = rankings[second][task_idx]
            if first_rank < second_rank:
                wins[first_idx] += 1.0
                losses[second_idx] += 1.0
            elif second_rank < first_rank:
                wins[second_idx] += 1.0
                losses[first_idx] += 1.0
            else:
                wins[first_idx] += 0.5
                losses[first_idx] += 0.5
                wins[second_idx] += 0.5
                losses[second_idx] += 0.5

    for row_idx in range(size):
        total_games = sum(comparisons[row_idx])
        matrix[row_idx][row_idx] = 2.0 + total_games
        for col_idx in range(size):
            if row_idx != col_idx:
                matrix[row_idx][col_idx] = -comparisons[row_idx][col_idx]

    vector = [1.0 + (wins[idx] - losses[idx]) / 2.0 for idx in range(size)]
    return matrix, vector


def linear_ordering_objective(order: Sequence[str], supports: dict[tuple[str, str], float]) -> float:
    total = 0.0
    for index, first in enumerate(order):
        for second in order[index + 1 :]:
            total += supports[(first, second)]
    return total


def linear_ordering_exact(
    algorithms: list[str],
    supports: dict[tuple[str, str], float],
) -> list[str]:
    best_order: tuple[str, ...] | None = None
    best_score: float | None = None
    for candidate in permutations(algorithms):
        score = linear_ordering_objective(candidate, supports)
        if best_score is None or score > best_score or (score == best_score and candidate < best_order):
            best_order = candidate
            best_score = score
    return list(best_order) if best_order is not None else algorithms


def linear_ordering_heuristic(
    rankings: Rankings,
    algorithms: list[str],
    supports: dict[tuple[str, str], float],
) -> list[str]:
    order = sorted(algorithms, key=lambda algorithm: (fmean(rankings[algorithm]), algorithm))
    best_score = linear_ordering_objective(order, supports)

    improved = True
    while improved:
        improved = False
        for left in range(len(order) - 1):
            for right in range(left + 1, len(order)):
                candidate = order[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_score = linear_ordering_objective(candidate, supports)
                if candidate_score > best_score:
                    order = candidate
                    best_score = candidate_score
                    improved = True
    return order


def polyrank_observations(
    algorithms: Sequence[str],
    supports: dict[tuple[str, str], float],
) -> list[dict[str, float | str]]:
    observations: list[dict[str, float | str]] = []
    for first, second in combinations(algorithms, 2):
        first_support = supports[(first, second)]
        second_support = supports[(second, first)]
        total = first_support + second_support
        if total <= 0:
            continue
        probability = first_support / total
        clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
        observations.append({"first": first, "second": second, "z": 2.0 * clipped - 1.0})
    return observations


def polyrank_transform(z_value: float, coefficients: Sequence[float]) -> float:
    total = 0.0
    for degree, coefficient in enumerate(coefficients):
        total += coefficient * (z_value ** (2 * degree + 1))
    return total


def least_squares_pairwise_scores(
    algorithms: Sequence[str],
    observations: Sequence[dict[str, float | str]],
    targets: Sequence[float],
) -> list[float]:
    size = len(algorithms)
    index = {algorithm: idx for idx, algorithm in enumerate(algorithms)}
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    vector = [0.0 for _ in range(size)]

    for observation, target in zip(observations, targets):
        first = str(observation["first"])
        second = str(observation["second"])
        first_idx = index[first]
        second_idx = index[second]
        matrix[first_idx][first_idx] += 1.0
        matrix[second_idx][second_idx] += 1.0
        matrix[first_idx][second_idx] -= 1.0
        matrix[second_idx][first_idx] -= 1.0
        vector[first_idx] += target
        vector[second_idx] -= target

    matrix[-1] = [1.0 for _ in range(size)]
    vector[-1] = 0.0
    return solve_linear_system(matrix, vector)


def fit_polyrank_coefficients(
    observations: Sequence[dict[str, float | str]],
    score_values: Sequence[float],
    algorithms: Sequence[str],
    degree: int,
) -> list[float]:
    basis_size = degree + 1
    matrix = [[0.0 for _ in range(basis_size)] for _ in range(basis_size)]
    vector = [0.0 for _ in range(basis_size)]
    index = {algorithm: idx for idx, algorithm in enumerate(algorithms)}

    for observation in observations:
        z_value = float(observation["z"])
        first_idx = index[str(observation["first"])]
        second_idx = index[str(observation["second"])]
        target = score_values[first_idx] - score_values[second_idx]
        basis = [z_value ** (2 * power + 1) for power in range(basis_size)]
        for row_idx in range(basis_size):
            vector[row_idx] += basis[row_idx] * target
            for col_idx in range(basis_size):
                matrix[row_idx][col_idx] += basis[row_idx] * basis[col_idx]

    ridge = 1e-6
    for idx in range(basis_size):
        matrix[idx][idx] += ridge
    return solve_linear_system(matrix, vector)


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(matrix)
    augmented = [row[:] + [vector[idx]] for idx, row in enumerate(matrix)]

    for pivot_idx in range(size):
        pivot_row = max(range(pivot_idx, size), key=lambda row_idx: abs(augmented[row_idx][pivot_idx]))
        pivot_value = augmented[pivot_row][pivot_idx]
        if abs(pivot_value) < 1e-12:
            raise ValueError("linear system is singular")
        if pivot_row != pivot_idx:
            augmented[pivot_idx], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_idx]

        pivot_value = augmented[pivot_idx][pivot_idx]
        for col_idx in range(pivot_idx, size + 1):
            augmented[pivot_idx][col_idx] /= pivot_value

        for row_idx in range(size):
            if row_idx == pivot_idx:
                continue
            factor = augmented[row_idx][pivot_idx]
            if factor == 0:
                continue
            for col_idx in range(pivot_idx, size + 1):
                augmented[row_idx][col_idx] -= factor * augmented[pivot_idx][col_idx]

    return [augmented[row_idx][size] for row_idx in range(size)]
