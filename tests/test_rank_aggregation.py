import unittest

from rank_aggregation.methods import (
    BestRankCountAggregator,
    FriedmanNemenyiRankAggregator,
    KemenyYoungAggregator,
    MeanQualityAggregator,
    MeanRankAggregator,
    MedianQualityAggregator,
    MedianRankAggregator,
    RescaledMeanQualityAggregator,
    ThresholdQualityAggregator,
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


if __name__ == "__main__":
    unittest.main()
