
## Implemented aggregation methods

### Curated tag vocabulary

**Input tags**

- `input:score`: consumes per-task quality or score values directly
- `input:rank`: consumes full rankings directly
- `input:pairwise`: relies on pairwise wins, losses, or preference margins

**Family tags**

- `family:central-tendency`: aggregates by a central statistic over tasks or rankings
- `family:positional`: uses position-dependent counts or weights
- `family:consensus`: seeks a ranking that best agrees with the input rankings overall
- `family:condorcet`: derived from pairwise majority relations and Condorcet-style reasoning
- `family:probabilistic`: based on an explicit probabilistic preference or ranking model
- `family:mcda`: belongs to multi-criteria decision analysis
- `family:outranking`: MCDA method based on outranking relations
- `family:benchmark-profile`: derived from benchmark profile methodology

**Operator tags**

- `operator:mean`: arithmetic mean is the main aggregation operator
- `operator:median`: median is the main aggregation operator
- `operator:geometric-mean`: geometric mean is the main aggregation operator
- `operator:harmonic-mean`: harmonic mean is the main aggregation operator

**Mechanism tags**

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

---

### Methods by group

1. **Score based**.    
   Methods that aggregate raw performance values or quality scores directly.
   
   a. `MeanQualityAggregator` (`Q-M`) - tags: `input:score`, `family:central-tendency`, `operator:mean`    
   b. `MedianQualityAggregator` (`Q-Md`) - tags: `input:score`, `family:central-tendency`, `operator:median`    
   c. `GeometricMeanQualityAggregator` (`Q-GM`) - tags: `input:score`, `family:central-tendency`, `operator:geometric-mean`    
   d. `HarmonicMeanQualityAggregator` (`Q-HM`) - tags: `input:score`, `family:central-tendency`, `operator:harmonic-mean`    
   e. `RescaledMeanQualityAggregator` (`Q-RM`) - tags: `input:score`, `family:central-tendency`, `operator:mean`, `mechanism:normalization`    
   f. `ThresholdQualityAggregator` (`Q-Th(theta)`) - tags: `input:score`, `mechanism:thresholding`

3. **Rank based**.    
   Methods that start from full rankings and combine them without reducing them to pairwise relations first.
   
   a. `MeanRankAggregator` (`R-M`) - tags: `input:rank`, `family:central-tendency`, `operator:mean`    
   b. `MedianRankAggregator` (`R-Md`) - tags: `input:rank`, `family:central-tendency`, `operator:median`    
   c. `BestRankCountAggregator` (`R-B`) - tags: `input:rank`, `family:positional`, `mechanism:counting`    
   d. `WorstRankCountAggregator` (`R-W`) - tags: `input:rank`, `family:positional`, `mechanism:counting`    
   e. `BordaCountAggregator` (`R-Borda`) - tags: `input:rank`, `family:positional`, `mechanism:scoring`    
   f. `DowdallHarmonicAggregator` (`R-Dowdall`) - tags: `input:rank`, `family:positional`, `mechanism:reciprocal-weighting`    
   g. `ReciprocalRankFusionAggregator` (`R-RRF`) - tags: `input:rank`, `family:positional`, `mechanism:reciprocal-weighting`    
   h. `KemenyYoungAggregator` (`R-Kem`) - tags: `input:rank`, `family:consensus`, `mechanism:optimization`

4. **Pairwise**.    
   Methods that derive the final ranking from pairwise comparisons, pairwise margins, or pairwise preference models.
   
   a. **Simple**.    
      Direct pairwise counting or margin-based aggregation rules.
      1. `CopelandPairwiseAggregator` (`R-Cop`) - tags: `input:pairwise`, `mechanism:counting`
      2. `MarginRowSumAggregator` (`R-MRS`) - tags: `input:pairwise`, `mechanism:pairwise-margins`, `mechanism:scoring`
   
   b. **Condorcet**.    
      Methods centered on majority comparisons and Condorcet-consistent reasoning.
      1. `MinimaxCondorcetAggregator` (`R-Minimax`) - tags: `input:pairwise`, `family:condorcet`, `mechanism:optimization`
      2. `RankedPairsTidemanAggregator` (`R-RP`) - tags: `input:pairwise`, `family:condorcet`
      3. `SchulzeBeatpathAggregator` (`R-Sch`) - tags: `input:pairwise`, `family:condorcet`
      4. `SplitCycleAggregator` (`R-SC`) - tags: `input:pairwise`, `family:condorcet`
      5. `RiverAggregator` (`R-River`) - tags: `input:pairwise`, `family:condorcet`
      6. `StableVotingAggregator` (`R-SV`) - tags: `input:pairwise`, `family:condorcet`
      7. `SimpleStableVotingAggregator` (`R-SSV`) - tags: `input:pairwise`, `family:condorcet`
   
   c. **Probabilistic**.    
      Pairwise models that estimate latent preferences through a statistical likelihood model.
      1. `BradleyTerryAggregator` (`R-BT`) - tags: `input:pairwise`, `family:probabilistic`, `mechanism:likelihood`
      2. `ThurstoneMostellerAggregator` (`R-TM`) - tags: `input:pairwise`, `family:probabilistic`, `mechanism:likelihood`
      3. `PolyRankAggregator` (`R-PolyRank`) - tags: `input:pairwise`, `family:probabilistic`, `mechanism:likelihood`
   
   d. **Stochastic**.    
      Methods that use random-walk or game-theoretic dynamics derived from pairwise preferences.
      1. `MarkovChainAggregator` (`R-MC`) - tags: `input:pairwise`, `mechanism:markov-chain`
      2. `MaximalLotteryAggregator` (`R-ML`) - tags: `input:pairwise`, `mechanism:game-theoretic`
   
   e. **Linear**.    
      Methods that recover ratings by solving a linear system built from pairwise outcomes.
      1. `MasseyRankingAggregator` (`R-Massey`) - tags: `input:pairwise`, `mechanism:linear-system`
      2. `ColleyRankingAggregator` (`R-Colley`) - tags: `input:pairwise`, `mechanism:linear-system`
   
   f. **Optimization**
      Methods defined primarily through an optimization objective over pairwise preferences.
      1. `LinearOrderingProblemAggregator` (`R-LOP`) - tags: `input:pairwise`, `mechanism:optimization`

5. **Probabilistic rank models**.    
   Methods that fit an explicit probability model over complete rankings or ranking generation.

   a. `PlackettLuceAggregator` (`R-PL`) - tags: `input:rank`, `family:probabilistic`, `mechanism:likelihood`

6. **Multi-criteria decision analysis (MCDA)**.    
   Methods from decision analysis that combine multiple criteria, often with preference modeling or outranking logic.
   
   a. `PROMETHEEIIAggregator` (`R-P2`) - tags: `input:score`, `family:mcda`, `family:outranking`    
   b. `ELECTREIIIAggregator` (`R-E3`) - tags: `input:score`, `family:mcda`, `family:outranking`, `mechanism:thresholding`    
   c. `TOPSISAggregator` (`R-TOPSIS`) - tags: `input:score`, `family:mcda`, `mechanism:normalization`    
   d. `VIKORAggregator` (`R-VIKOR`) - tags: `input:score`, `family:mcda`, `mechanism:normalization`    

7. **Benchmark profiles**.    
   Methods designed around benchmark-wide comparison procedures such as statistical comparison protocols or performance profiles.
   
   a. `FriedmanNemenyiRankAggregator` (`R-N`) - tags: `input:rank`, `family:benchmark-profile`, `mechanism:statistical-test`    
   b. `DMAUCPerformanceProfileAggregator` (`DM-AUC`) - tags: `input:score`, `family:benchmark-profile`    
   c. `DMLBOLeaveOneOutProfileAggregator` (`DM-LBO`) - tags: `input:score`, `family:benchmark-profile`    


### Methods found somewhere, but not implemented as separate classes

The following methods are intentionally not exposed as standalone implementations:

- `Plurality Vote`
  - Conceptually equivalent to `R-B` (`BestRankCountAggregator`), but under a different name.
- `Normalized Score Aggregation`
  - Close in spirit to `Q-RM` (`RescaledMeanQualityAggregator`), but not identical:
    `Normalized Score Aggregation` uses `L_t/H_t` scaling, while `Q-RM` uses per-task min-max scaling over the observed algorithms.
