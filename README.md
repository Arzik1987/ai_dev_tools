# ai_dev_tools

## Implemented aggregation methods

- `MeanRankAggregator` (`R-M`)
- `MedianRankAggregator` (`R-Md`)
- `MeanQualityAggregator` (`Q-M`)
- `MedianQualityAggregator` (`Q-Md`)
- `GeometricMeanQualityAggregator` (`Q-GM`)
- `HarmonicMeanQualityAggregator` (`Q-HM`)
- `RescaledMeanQualityAggregator` (`Q-RM`)
- `BestRankCountAggregator` (`R-B`)
- `WorstRankCountAggregator` (`R-W`)
- `BordaCountAggregator` (`R-Borda`)
- `DowdallHarmonicAggregator` (`R-Dowdall`)
- `ReciprocalRankFusionAggregator` (`R-RRF`)
- `MarginRowSumAggregator` (`R-MRS`)
- `MinimaxCondorcetAggregator` (`R-Minimax`)
- `RankedPairsTidemanAggregator` (`R-RP`)
- `SchulzeBeatpathAggregator` (`R-Sch`)
- `SplitCycleAggregator` (`R-SC`)
- `RiverAggregator` (`R-River`)
- `PlackettLuceAggregator` (`R-PL`)
- `MarkovChainAggregator` (`R-MC`)
- `MaximalLotteryAggregator` (`R-ML`)
- `BradleyTerryAggregator` (`R-BT`)
- `ThurstoneMostellerAggregator` (`R-TM`)
- `StableVotingAggregator` (`R-SV`)
- `SimpleStableVotingAggregator` (`R-SSV`)
- `CopelandPairwiseAggregator` (`R-Cop`)
- `DMAUCPerformanceProfileAggregator` (`DM-AUC`)
- `DMLBOLeaveOneOutProfileAggregator` (`DM-LBO`)
- `ThresholdQualityAggregator` (`Q-Th(theta)`)
- `FriedmanNemenyiRankAggregator` (`R-N`)
- `KemenyYoungAggregator` (`R-Kem`)

## Comparison with `all_methods.md`

### Overlap (in both code and `all_methods.md`)

- `R-M` (Mean Rank)
- `R-Md` (Median Rank)
- `Q-M` (Mean Quality)
- `Q-Md` (Median Quality)
- `Q-GM` (Geometric Mean of Performance)
- `Q-HM` (Harmonic Mean of Performance)
- `Q-RM` (Rescaled Mean Quality)
- `R-B` (Best Rank Count)
- `R-W` (Worst Rank Count)
- `Q-Th` (Threshold Quality)
- `R-N` (Friedman-Nemenyi)
- `R-Kem` (Kemeny-Young)
- `R-Borda` (Borda Count)
- `R-Dowdall` (Dowdall Harmonic Rule)
- `R-RRF` (Reciprocal Rank Fusion)
- `R-MRS` (Margin-based Pairwise Row-Sum)
- `R-Minimax` (Minimax Condorcet)
- `R-RP` (Ranked Pairs Tideman)
- `R-Sch` (Schulze / Beatpath)
- `R-SC` (Split Cycle)
- `R-River` (River)
- `R-PL` (Plackett-Luce Model)
- `R-MC` (Markov Chain Aggregation)
- `R-ML` (Maximal Lottery)
- `R-BT` (Bradley-Terry)
- `R-TM` (Thurstone-Mosteller)
- `R-SV` (Stable Voting)
- `R-SSV` (Simple Stable Voting)
- `R-Cop` (Copeland Pairwise)
- `DM-AUC` (Dolan-More Performance Profiles)
- `DM-LBO` (Iterative Leave-One-Out Profile Ranking)

### In `all_methods.md` but not implemented in code

- Plurality Vote
- Normalized Score Aggregation

### Naming/content mismatches

- `all_methods.md` has a typo: `sMean Rank (R-M)` (leading `s`).
- `all_methods.md` labels Friedman-Nemenyi as `R-N_p-value`; code uses `R-N` with configurable `alpha`.
- `Plurality Vote` is conceptually equivalent to `R-B` (Best Rank Count), but named differently.
- `Normalized Score Aggregation` is conceptually close to `Q-RM`, but definition differs (`L_t/H_t` scaling vs per-task min-max over observed algorithms).
