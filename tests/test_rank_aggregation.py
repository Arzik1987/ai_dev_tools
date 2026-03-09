import unittest

from rank_aggregation.methods import (
    BestRankCountAggregator,
    BradleyTerryAggregator,
    BordaCountAggregator,
    CopelandPairwiseAggregator,
    ColleyRankingAggregator,
    DMAUCPerformanceProfileAggregator,
    DMLBOLeaveOneOutProfileAggregator,
    DowdallHarmonicAggregator,
    ELECTREIIIAggregator,
    FriedmanNemenyiRankAggregator,
    GeometricMeanQualityAggregator,
    HarmonicMeanQualityAggregator,
    KemenyYoungAggregator,
    LinearOrderingProblemAggregator,
    MarginRowSumAggregator,
    MasseyRankingAggregator,
    MarkovChainAggregator,
    MaximalLotteryAggregator,
    MeanQualityAggregator,
    MeanRankAggregator,
    MedianQualityAggregator,
    MedianRankAggregator,
    MinimaxCondorcetAggregator,
    PlackettLuceAggregator,
    PolyRankAggregator,
    PROMETHEEIIAggregator,
    RankedPairsTidemanAggregator,
    ReciprocalRankFusionAggregator,
    RescaledMeanQualityAggregator,
    RiverAggregator,
    SchulzeBeatpathAggregator,
    SimpleStableVotingAggregator,
    SplitCycleAggregator,
    StableVotingAggregator,
    TOPSISAggregator,
    ThresholdQualityAggregator,
    ThurstoneMostellerAggregator,
    VIKORAggregator,
    WorstRankCountAggregator,
)


class TestRankMethods(unittest.TestCase):
    def test_mean_rank(self) -> None:
        rankings = {
            "algo_a": [1, 2, 2],
            "algo_b": [2, 1, 3],
            "algo_c": [3, 3, 1],
        }
        result = MeanRankAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 5 / 3)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_median_rank_even(self) -> None:
        rankings = {
            "algo_a": [1, 4, 2, 2],
            "algo_b": [2, 3, 3, 3],
        }
        result = MedianRankAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 2.0)
        self.assertEqual(result.ranking, ["algo_a", "algo_b"])

    def test_best_rank_count(self) -> None:
        rankings = {
            "algo_a": [1, 2, 1, 3],
            "algo_b": [2, 1, 3, 1],
            "algo_c": [3, 3, 2, 2],
        }
        result = BestRankCountAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 2.0)
        self.assertEqual(result.scores["algo_b"], 2.0)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_worst_rank_count(self) -> None:
        rankings = {
            "algo_a": [1, 2, 3, 3],
            "algo_b": [2, 3, 1, 2],
            "algo_c": [3, 1, 2, 1],
        }
        result = WorstRankCountAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 2.0)
        self.assertEqual(result.scores["algo_b"], 1.0)
        self.assertEqual(result.scores["algo_c"], 1.0)
        self.assertEqual(result.ranking, ["algo_b", "algo_c", "algo_a"])

    def test_borda_count(self) -> None:
        rankings = {
            "algo_a": [1, 2, 2],
            "algo_b": [2, 1, 3],
            "algo_c": [3, 3, 1],
        }
        result = BordaCountAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 4.0)
        self.assertEqual(result.scores["algo_b"], 3.0)
        self.assertEqual(result.scores["algo_c"], 2.0)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_dowdall_harmonic(self) -> None:
        rankings = {
            "algo_a": [1, 2, 2],
            "algo_b": [2, 1, 3],
            "algo_c": [3, 3, 1],
        }
        result = DowdallHarmonicAggregator().aggregate(rankings)
        self.assertAlmostEqual(result.scores["algo_a"], 2.0)
        self.assertAlmostEqual(result.scores["algo_b"], 1.8333333333333333)
        self.assertAlmostEqual(result.scores["algo_c"], 1.6666666666666665)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_reciprocal_rank_fusion(self) -> None:
        rankings = {
            "algo_a": [1, 2, 3],
            "algo_b": [2, 1, 2],
            "algo_c": [3, 3, 1],
        }
        result = ReciprocalRankFusionAggregator(k=10).aggregate(rankings)
        self.assertAlmostEqual(result.scores["algo_a"], 0.2511655011655012)
        self.assertAlmostEqual(result.scores["algo_b"], 0.25757575757575757)
        self.assertAlmostEqual(result.scores["algo_c"], 0.24475524475524474)
        self.assertEqual(result.ranking, ["algo_b", "algo_a", "algo_c"])

    def test_minimax_condorcet(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 2, 3],
            "algo_b": [2, 2, 1, 3, 1],
            "algo_c": [3, 3, 3, 1, 2],
        }
        result = MinimaxCondorcetAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 2.0)
        self.assertEqual(result.scores["algo_b"], 3.0)
        self.assertEqual(result.scores["algo_c"], 4.0)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_ranked_pairs_tideman(self) -> None:
        rankings = {
            "algo_a": [1, 1, 3, 3, 2],
            "algo_b": [2, 2, 1, 1, 3],
            "algo_c": [3, 3, 2, 2, 1],
        }
        result = RankedPairsTidemanAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_ranked_pairs_condorcet_winner_first(self) -> None:
        rankings = {
            "algo_a": [1, 1, 1, 2, 2],
            "algo_b": [2, 2, 3, 1, 3],
            "algo_c": [3, 3, 2, 3, 1],
        }
        result = RankedPairsTidemanAggregator().aggregate(rankings)
        self.assertEqual(result.ranking[0], "algo_a")

    def test_schulze_beatpath(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = SchulzeBeatpathAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 2.0)
        self.assertEqual(result.scores["algo_b"], 1.0)
        self.assertEqual(result.scores["algo_c"], 0.0)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_margin_row_sum(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = MarginRowSumAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 6.0)
        self.assertEqual(result.scores["algo_b"], 4.0)
        self.assertEqual(result.scores["algo_c"], -10.0)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_split_cycle(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = SplitCycleAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])

    def test_river(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = RiverAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_markov_chain_aggregation(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = MarkovChainAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertAlmostEqual(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])

    def test_maximal_lottery_condorcet(self) -> None:
        rankings = {
            "algo_a": [1, 1, 1, 2, 2],
            "algo_b": [2, 2, 3, 1, 3],
            "algo_c": [3, 3, 2, 3, 1],
        }
        result = MaximalLotteryAggregator().aggregate(rankings)
        self.assertEqual(result.ranking[0], "algo_a")
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_c"])


class TestQualityMethods(unittest.TestCase):
    def test_mean_quality(self) -> None:
        qualities = {
            "algo_a": [0.90, 0.80, 0.85],
            "algo_b": [0.88, 0.81, 0.84],
        }
        result = MeanQualityAggregator().aggregate(qualities)
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertEqual(result.ranking, ["algo_a", "algo_b"])

    def test_median_quality(self) -> None:
        qualities = {
            "algo_a": [0.40, 0.85, 0.86],
            "algo_b": [0.70, 0.75, 0.80],
        }
        result = MedianQualityAggregator().aggregate(qualities)
        self.assertEqual(result.scores["algo_a"], 0.85)
        self.assertEqual(result.ranking, ["algo_a", "algo_b"])

    def test_geometric_mean_quality(self) -> None:
        qualities = {
            "algo_a": [0.8, 0.9, 1.0],
            "algo_b": [0.9, 0.9, 0.9],
            "algo_c": [1.0, 1.0, 0.7],
        }
        result = GeometricMeanQualityAggregator().aggregate(qualities)
        self.assertAlmostEqual(result.scores["algo_a"], 0.896280949311433)
        self.assertAlmostEqual(result.scores["algo_b"], 0.9)
        self.assertAlmostEqual(result.scores["algo_c"], 0.8879040017426006)
        self.assertEqual(result.ranking, ["algo_b", "algo_a", "algo_c"])

    def test_harmonic_mean_quality(self) -> None:
        qualities = {
            "algo_a": [0.8, 0.9, 1.0],
            "algo_b": [0.9, 0.9, 0.9],
            "algo_c": [1.0, 1.0, 0.7],
        }
        result = HarmonicMeanQualityAggregator().aggregate(qualities)
        self.assertAlmostEqual(result.scores["algo_a"], 0.8925619834710744)
        self.assertAlmostEqual(result.scores["algo_b"], 0.9)
        self.assertAlmostEqual(result.scores["algo_c"], 0.875)
        self.assertEqual(result.ranking, ["algo_b", "algo_a", "algo_c"])

    def test_rescaled_mean_quality(self) -> None:
        qualities = {
            "algo_a": [80, 0.80],
            "algo_b": [100, 0.70],
            "algo_c": [60, 0.60],
        }
        result = RescaledMeanQualityAggregator().aggregate(qualities)
        self.assertAlmostEqual(result.scores["algo_a"], 0.75)
        self.assertAlmostEqual(result.scores["algo_b"], 0.75)
        self.assertAlmostEqual(result.scores["algo_c"], 0.0)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_threshold_quality_count(self) -> None:
        qualities = {
            "algo_a": [0.90, 0.88, 0.70],
            "algo_b": [0.91, 0.80, 0.69],
            "algo_c": [0.85, 0.87, 0.71],
        }
        result = ThresholdQualityAggregator(theta=0.95).aggregate(qualities)
        self.assertEqual(result.scores["algo_a"], 3.0)
        self.assertEqual(result.scores["algo_b"], 2.0)
        self.assertEqual(result.scores["algo_c"], 2.0)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_threshold_quality_normalized(self) -> None:
        qualities = {
            "algo_a": [10, 9],
            "algo_b": [9, 10],
        }
        result = ThresholdQualityAggregator(theta=0.9, normalize=True).aggregate(qualities)
        self.assertEqual(result.scores["algo_a"], 1.0)
        self.assertEqual(result.scores["algo_b"], 1.0)

    def test_promethee_ii(self) -> None:
        qualities = {
            "algo_a": [0.95, 0.92, 0.90],
            "algo_b": [0.91, 0.88, 0.89],
            "algo_c": [0.80, 0.85, 0.86],
        }
        result = PROMETHEEIIAggregator().aggregate(qualities)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])

    def test_electre_iii(self) -> None:
        qualities = {
            "algo_a": [0.95, 0.92, 0.90],
            "algo_b": [0.91, 0.88, 0.89],
            "algo_c": [0.80, 0.85, 0.86],
        }
        result = ELECTREIIIAggregator(q=0.01, p=0.04, v=0.10).aggregate(qualities)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])

    def test_topsis(self) -> None:
        qualities = {
            "algo_a": [0.95, 0.92, 0.90],
            "algo_b": [0.91, 0.88, 0.89],
            "algo_c": [0.80, 0.85, 0.86],
        }
        result = TOPSISAggregator().aggregate(qualities)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])

    def test_vikor(self) -> None:
        qualities = {
            "algo_a": [0.95, 0.92, 0.90],
            "algo_b": [0.91, 0.88, 0.89],
            "algo_c": [0.80, 0.85, 0.86],
        }
        result = VIKORAggregator().aggregate(qualities)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertLess(result.scores["algo_a"], result.scores["algo_b"])
        self.assertLess(result.scores["algo_b"], result.scores["algo_c"])


class TestAdvancedMethods(unittest.TestCase):
    def test_friedman_nemenyi_significant(self) -> None:
        rankings = {
            "algo_a": [1, 1, 1, 1, 1],
            "algo_b": [2, 2, 2, 2, 2],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = FriedmanNemenyiRankAggregator(alpha=0.05).aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_friedman_nemenyi_not_significant(self) -> None:
        rankings = {
            "algo_a": [1, 2, 3],
            "algo_b": [2, 3, 1],
            "algo_c": [3, 1, 2],
        }
        result = FriedmanNemenyiRankAggregator(alpha=0.05).aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_kemeny_young(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2],
            "algo_b": [2, 3, 1],
            "algo_c": [3, 2, 3],
        }
        result = KemenyYoungAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_linear_ordering_problem(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1],
            "algo_b": [2, 2, 1, 2],
            "algo_c": [3, 3, 3, 3],
        }
        result = LinearOrderingProblemAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_polyrank(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1],
            "algo_b": [2, 2, 1, 2],
            "algo_c": [3, 3, 3, 3],
        }
        result = PolyRankAggregator(degree=2).aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])

    def test_bradley_terry(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = BradleyTerryAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])

    def test_thurstone_mosteller(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = ThurstoneMostellerAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])

    def test_stable_voting(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = StableVotingAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_simple_stable_voting(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = SimpleStableVotingAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_plackett_luce(self) -> None:
        rankings = {
            "algo_a": [1, 1, 1, 2, 2],
            "algo_b": [2, 2, 3, 1, 3],
            "algo_c": [3, 3, 2, 3, 1],
        }
        result = PlackettLuceAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertAlmostEqual(result.scores["algo_b"], result.scores["algo_c"])

    def test_copeland_pairwise(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1, 2],
            "algo_b": [2, 2, 1, 2, 1],
            "algo_c": [3, 3, 3, 3, 3],
        }
        result = CopelandPairwiseAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 2.0)
        self.assertEqual(result.scores["algo_b"], 1.0)
        self.assertEqual(result.scores["algo_c"], 0.0)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])

    def test_copeland_pairwise_tied_duel(self) -> None:
        rankings = {
            "algo_a": [1, 2],
            "algo_b": [2, 1],
        }
        result = CopelandPairwiseAggregator().aggregate(rankings)
        self.assertEqual(result.scores["algo_a"], 0.5)
        self.assertEqual(result.scores["algo_b"], 0.5)
        self.assertEqual(result.ranking, ["algo_a", "algo_b"])

    def test_dm_auc_profile(self) -> None:
        rankings = {
            "algo_a": [1, 1, 10],
            "algo_b": [2, 2, 2],
            "algo_c": [3, 3, 1],
        }
        result = DMAUCPerformanceProfileAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_b", "algo_c", "algo_a"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])
        self.assertGreater(result.scores["algo_c"], result.scores["algo_a"])

    def test_dm_lbo_leave_one_out(self) -> None:
        rankings = {
            "algo_a": [1, 1, 10],
            "algo_b": [2, 2, 2],
            "algo_c": [3, 3, 1],
        }
        result = DMLBOLeaveOneOutProfileAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_b", "algo_c", "algo_a"])
        self.assertEqual(result.scores["algo_b"], 3.0)
        self.assertEqual(result.scores["algo_c"], 2.0)
        self.assertEqual(result.scores["algo_a"], 1.0)

    def test_dm_lbo_tie_break(self) -> None:
        rankings = {
            "algo_a": [1, 2],
            "algo_b": [2, 1],
        }
        result = DMLBOLeaveOneOutProfileAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b"])

    def test_massey_ranking(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1],
            "algo_b": [2, 2, 1, 2],
            "algo_c": [3, 3, 3, 3],
        }
        result = MasseyRankingAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])

    def test_colley_ranking(self) -> None:
        rankings = {
            "algo_a": [1, 1, 2, 1],
            "algo_b": [2, 2, 1, 2],
            "algo_c": [3, 3, 3, 3],
        }
        result = ColleyRankingAggregator().aggregate(rankings)
        self.assertEqual(result.ranking, ["algo_a", "algo_b", "algo_c"])
        self.assertGreater(result.scores["algo_a"], result.scores["algo_b"])
        self.assertGreater(result.scores["algo_b"], result.scores["algo_c"])


class TestValidation(unittest.TestCase):
    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            MeanRankAggregator().aggregate({})

    def test_empty_rank_list_raises(self) -> None:
        with self.assertRaises(ValueError):
            MeanRankAggregator().aggregate({"algo_a": [], "algo_b": []})

    def test_mismatched_task_counts_raise(self) -> None:
        with self.assertRaises(ValueError):
            MeanRankAggregator().aggregate({"algo_a": [1, 2], "algo_b": [1]})

    def test_invalid_theta_raises(self) -> None:
        with self.assertRaises(ValueError):
            ThresholdQualityAggregator(theta=1.5)

    def test_invalid_bradley_terry_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            BradleyTerryAggregator(max_iter=0)
        with self.assertRaises(ValueError):
            BradleyTerryAggregator(tol=0.0)

    def test_invalid_thurstone_mosteller_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            ThurstoneMostellerAggregator(max_iter=0)
        with self.assertRaises(ValueError):
            ThurstoneMostellerAggregator(tol=0.0)
        with self.assertRaises(ValueError):
            ThurstoneMostellerAggregator(ridge=-1.0)
        with self.assertRaises(ValueError):
            ThurstoneMostellerAggregator(initial_step=0.0)

    def test_invalid_plackett_luce_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            PlackettLuceAggregator(max_iter=0)
        with self.assertRaises(ValueError):
            PlackettLuceAggregator(tol=0.0)

    def test_invalid_markov_chain_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            MarkovChainAggregator(damping=1.0)
        with self.assertRaises(ValueError):
            MarkovChainAggregator(max_iter=0)
        with self.assertRaises(ValueError):
            MarkovChainAggregator(tol=0.0)

    def test_invalid_maximal_lottery_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            MaximalLotteryAggregator(max_iter=0)
        with self.assertRaises(ValueError):
            MaximalLotteryAggregator(tol=0.0)

    def test_invalid_dm_auc_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            DMAUCPerformanceProfileAggregator(tau_max=1.0)
        with self.assertRaises(ValueError):
            DMAUCPerformanceProfileAggregator().aggregate({"algo_a": [0], "algo_b": [1]})

    def test_invalid_dm_lbo_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            DMLBOLeaveOneOutProfileAggregator(tau_max=1.0)
        with self.assertRaises(ValueError):
            DMLBOLeaveOneOutProfileAggregator().aggregate({"algo_a": [0], "algo_b": [1]})

    def test_invalid_dowdall_ranks_raise(self) -> None:
        with self.assertRaises(ValueError):
            DowdallHarmonicAggregator().aggregate({"algo_a": [1, 0], "algo_b": [2, 1]})

    def test_invalid_geometric_mean_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            GeometricMeanQualityAggregator().aggregate({"algo_a": [0.8, -0.1], "algo_b": [0.7, 0.2]})

    def test_invalid_harmonic_mean_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            HarmonicMeanQualityAggregator().aggregate({"algo_a": [0.8, 0.0], "algo_b": [0.7, 0.2]})

    def test_invalid_rrf_params_or_ranks_raise(self) -> None:
        with self.assertRaises(ValueError):
            ReciprocalRankFusionAggregator(k=-1)
        with self.assertRaises(ValueError):
            ReciprocalRankFusionAggregator().aggregate({"algo_a": [1, 0], "algo_b": [2, 1]})

    def test_invalid_vikor_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            VIKORAggregator(v=1.1)

    def test_invalid_mcdm_params_raise(self) -> None:
        qualities = {"algo_a": [1.0, 2.0], "algo_b": [2.0, 1.0]}
        with self.assertRaises(ValueError):
            PROMETHEEIIAggregator(weights=[1.0]).aggregate(qualities)
        with self.assertRaises(ValueError):
            ELECTREIIIAggregator(q=0.2, p=0.1).aggregate(qualities)
        with self.assertRaises(ValueError):
            ELECTREIIIAggregator(p=0.3, v=0.2).aggregate(qualities)

    def test_invalid_lop_and_polyrank_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            LinearOrderingProblemAggregator(exact_max_algorithms=1)
        with self.assertRaises(ValueError):
            PolyRankAggregator(degree=-1)
        with self.assertRaises(ValueError):
            PolyRankAggregator(max_iter=0)
        with self.assertRaises(ValueError):
            PolyRankAggregator(tol=0.0)


if __name__ == "__main__":
    unittest.main()
