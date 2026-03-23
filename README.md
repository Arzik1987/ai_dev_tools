
## Implemented aggregation methods

### Curated tag vocabulary

- `input:score`: consumes per-task quality or score values directly
- `input:rank`: consumes full rankings directly
- `input:pairwise`: relies on pairwise wins, losses, or preference margins
- `family:central-tendency`: aggregates by a central statistic over tasks or rankings
- `family:positional`: uses position-dependent counts or weights
- `family:consensus`: seeks a ranking that best agrees with the input rankings overall
- `family:condorcet`: derived from pairwise majority relations and Condorcet-style reasoning
- `family:probabilistic`: based on an explicit probabilistic preference or ranking model
- `family:mcda`: belongs to multi-criteria decision analysis
- `family:outranking`: MCDA method based on outranking relations
- `family:benchmark-profile`: derived from benchmark profile methodology
- `operator:mean`: arithmetic mean is the main aggregation operator
- `operator:median`: median is the main aggregation operator
- `operator:geometric-mean`: geometric mean is the main aggregation operator
- `operator:harmonic-mean`: harmonic mean is the main aggregation operator
- `mechanism:normalization`: normalizes or rescales raw scores before aggregation
- `mechanism:thresholding`: applies explicit cutoffs or acceptance thresholds
- `mechanism:counting`: aggregates by counts of rank or preference events
- `mechanism:scoring`: aggregates by additive position or preference scores
- `mechanism:reciprocal-weighting`: uses reciprocal position-based weighting
- `mechanism:pairwise-margins`: uses pairwise win/loss margins directly
- `mechanism:linear-system`: solves a linear rating system
- `mechanism:optimization`: defined through an optimization objective
- `mechanism:markov-chain`: derives rankings from a Markov chain or stationary distribution
- `mechanism:game-theoretic`: derived from a game-theoretic solution concept
- `mechanism:likelihood`: fitted through a likelihood-based statistical model
- `mechanism:statistical-test`: uses statistical hypothesis testing

- `score_based`
  - `MeanQualityAggregator` (`Q-M`) - tags: `input:score`, `family:central-tendency`, `operator:mean`
  - `MedianQualityAggregator` (`Q-Md`) - tags: `input:score`, `family:central-tendency`, `operator:median`
  - `GeometricMeanQualityAggregator` (`Q-GM`) - tags: `input:score`, `family:central-tendency`, `operator:geometric-mean`
  - `HarmonicMeanQualityAggregator` (`Q-HM`) - tags: `input:score`, `family:central-tendency`, `operator:harmonic-mean`
  - `RescaledMeanQualityAggregator` (`Q-RM`) - tags: `input:score`, `family:central-tendency`, `operator:mean`, `mechanism:normalization`
  - `ThresholdQualityAggregator` (`Q-Th(theta)`) - tags: `input:score`, `mechanism:thresholding`

- `rank_based`
  - `MeanRankAggregator` (`R-M`) - tags: `input:rank`, `family:central-tendency`, `operator:mean`
  - `MedianRankAggregator` (`R-Md`) - tags: `input:rank`, `family:central-tendency`, `operator:median`
  - `BestRankCountAggregator` (`R-B`) - tags: `input:rank`, `family:positional`, `mechanism:counting`
  - `WorstRankCountAggregator` (`R-W`) - tags: `input:rank`, `family:positional`, `mechanism:counting`
  - `BordaCountAggregator` (`R-Borda`) - tags: `input:rank`, `family:positional`, `mechanism:scoring`
  - `DowdallHarmonicAggregator` (`R-Dowdall`) - tags: `input:rank`, `family:positional`, `mechanism:reciprocal-weighting`
  - `ReciprocalRankFusionAggregator` (`R-RRF`) - tags: `input:rank`, `family:positional`, `mechanism:reciprocal-weighting`
  - `KemenyYoungAggregator` (`R-Kem`) - tags: `input:rank`, `family:consensus`, `mechanism:optimization`

- `pairwise`
  - `simple`
    - `CopelandPairwiseAggregator` (`R-Cop`) - tags: `input:pairwise`, `mechanism:counting`
    - `MarginRowSumAggregator` (`R-MRS`) - tags: `input:pairwise`, `mechanism:pairwise-margins`, `mechanism:scoring`
  - `condorcet`
    - `MinimaxCondorcetAggregator` (`R-Minimax`) - tags: `input:pairwise`, `family:condorcet`, `mechanism:optimization`
    - `RankedPairsTidemanAggregator` (`R-RP`) - tags: `input:pairwise`, `family:condorcet`
    - `SchulzeBeatpathAggregator` (`R-Sch`) - tags: `input:pairwise`, `family:condorcet`
    - `SplitCycleAggregator` (`R-SC`) - tags: `input:pairwise`, `family:condorcet`
    - `RiverAggregator` (`R-River`) - tags: `input:pairwise`, `family:condorcet`
    - `StableVotingAggregator` (`R-SV`) - tags: `input:pairwise`, `family:condorcet`
    - `SimpleStableVotingAggregator` (`R-SSV`) - tags: `input:pairwise`, `family:condorcet`
  - `probabilistic`
    - `BradleyTerryAggregator` (`R-BT`) - tags: `input:pairwise`, `family:probabilistic`, `mechanism:likelihood`
    - `ThurstoneMostellerAggregator` (`R-TM`) - tags: `input:pairwise`, `family:probabilistic`, `mechanism:likelihood`
    - `PolyRankAggregator` (`R-PolyRank`) - tags: `input:pairwise`, `family:probabilistic`, `mechanism:likelihood`
  - `stochastic`
    - `MarkovChainAggregator` (`R-MC`) - tags: `input:pairwise`, `mechanism:markov-chain`
    - `MaximalLotteryAggregator` (`R-ML`) - tags: `input:pairwise`, `mechanism:game-theoretic`
  - `linear`
    - `MasseyRankingAggregator` (`R-Massey`) - tags: `input:pairwise`, `mechanism:linear-system`
    - `ColleyRankingAggregator` (`R-Colley`) - tags: `input:pairwise`, `mechanism:linear-system`
  - `optimization`
    - `LinearOrderingProblemAggregator` (`R-LOP`) - tags: `input:pairwise`, `mechanism:optimization`

- `probabilistic_rank_models`
  - `PlackettLuceAggregator` (`R-PL`) - tags: `input:rank`, `family:probabilistic`, `mechanism:likelihood`

- `mcda`
  - `PROMETHEEIIAggregator` (`R-P2`) - tags: `input:score`, `family:mcda`, `family:outranking`
  - `ELECTREIIIAggregator` (`R-E3`) - tags: `input:score`, `family:mcda`, `family:outranking`, `mechanism:thresholding`
  - `TOPSISAggregator` (`R-TOPSIS`) - tags: `input:score`, `family:mcda`, `mechanism:normalization`
  - `VIKORAggregator` (`R-VIKOR`) - tags: `input:score`, `family:mcda`, `mechanism:normalization`

- `benchmark_profiles`
  - `FriedmanNemenyiRankAggregator` (`R-N`) - tags: `input:rank`, `family:benchmark-profile`, `mechanism:statistical-test`
  - `DMAUCPerformanceProfileAggregator` (`DM-AUC`) - tags: `input:score`, `family:benchmark-profile`
  - `DMLBOLeaveOneOutProfileAggregator` (`DM-LBO`) - tags: `input:score`, `family:benchmark-profile`


### Methods found somewhere, but not implemented as separate classes

The following methods are intentionally not exposed as standalone implementations:

- `Plurality Vote`
  - Conceptually equivalent to `R-B` (`BestRankCountAggregator`), but under a different name.
- `Normalized Score Aggregation`
  - Close in spirit to `Q-RM` (`RescaledMeanQualityAggregator`), but not identical:
    `Normalized Score Aggregation` uses `L_t/H_t` scaling, while `Q-RM` uses per-task min-max scaling over the observed algorithms.
