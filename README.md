
## Implemented aggregation methods

### Methods by group

1. **Score based**.    
   Methods that aggregate raw performance values or quality scores directly.
   
   a. **MeanQualityAggregator** (*Q-M*)    
      tags: `input:score`, `family:central-tendency`, `operator:mean`, `solver:direct-statistic`, `complexity:polytime`, `maturity:standard`, `domain:benchmarking`, `assumption:ties-irrelevant`    
   b. **MedianQualityAggregator** (*Q-Md*)    
      tags: `input:score`, `family:central-tendency`, `operator:median`, `solver:direct-statistic`, `complexity:polytime`, `maturity:standard`, `domain:benchmarking`, `assumption:ties-irrelevant`    
   c. **GeometricMeanQualityAggregator** (*Q-GM*)    
      tags: `input:score`, `family:central-tendency`, `operator:geometric-mean`, `solver:direct-statistic`, `complexity:polytime`, `maturity:standard`, `domain:benchmarking`, `assumption:ties-irrelevant`, `assumption:nonnegative-scores`    
   d. **HarmonicMeanQualityAggregator** (*Q-HM*)    
      tags: `input:score`, `family:central-tendency`, `operator:harmonic-mean`, `solver:direct-statistic`, `complexity:polytime`, `maturity:standard`, `domain:benchmarking`, `assumption:ties-irrelevant`, `assumption:positive-scores`    
   e. **RescaledMeanQualityAggregator** (*Q-RM*)    
      tags: `input:score`, `family:central-tendency`, `operator:mean`, `mechanism:normalization`, `solver:direct-statistic`, `complexity:polytime`, `maturity:standard`, `domain:benchmarking`, `assumption:ties-irrelevant`    
   f. **ThresholdQualityAggregator** (*Q-Th(theta)*)    
      tags: `input:score`, `mechanism:thresholding`, `solver:direct-counting`, `complexity:polytime`, `maturity:specialized`, `domain:benchmarking`, `assumption:ties-irrelevant`

3. **Rank based**.    
   Methods that start from full rankings and combine them without reducing them to pairwise relations first.
   
   a. **MeanRankAggregator** (*R-M*)    
      tags: `input:rank`, `family:central-tendency`, `operator:mean`, `solver:direct-statistic`, `complexity:polytime`, `maturity:standard`, `domain:benchmarking`, `assumption:complete-rankings`, `assumption:strict-ranks`    
   b. **MedianRankAggregator** (*R-Md*)    
      tags: `input:rank`, `family:central-tendency`, `operator:median`, `solver:direct-statistic`, `complexity:polytime`, `maturity:standard`, `domain:benchmarking`, `assumption:complete-rankings`, `assumption:strict-ranks`    
   c. **BestRankCountAggregator** (*R-B*)    
      tags: `input:rank`, `family:positional`, `mechanism:counting`, `solver:direct-counting`, `complexity:polytime`, `maturity:classical`, `domain:social-choice`, `assumption:complete-rankings`, `assumption:strict-ranks`    
   d. **WorstRankCountAggregator** (*R-W*)    
      tags: `input:rank`, `family:positional`, `mechanism:counting`, `solver:direct-counting`, `complexity:polytime`, `maturity:specialized`, `domain:benchmarking`, `assumption:complete-rankings`, `assumption:strict-ranks`    
   e. **BordaCountAggregator** (*R-Borda*)    
      tags: `input:rank`, `family:positional`, `mechanism:scoring`, `solver:direct-scoring`, `complexity:polytime`, `maturity:classical`, `domain:social-choice`, `assumption:complete-rankings`, `assumption:strict-ranks`    
   f. **DowdallHarmonicAggregator** (*R-Dowdall*)    
      tags: `input:rank`, `family:positional`, `mechanism:reciprocal-weighting`, `solver:direct-scoring`, `complexity:polytime`, `maturity:classical`, `domain:social-choice`, `assumption:complete-rankings`, `assumption:strict-ranks`, `assumption:strict-positive-ranks`    
   g. **ReciprocalRankFusionAggregator** (*R-RRF*)    
      tags: `input:rank`, `family:positional`, `mechanism:reciprocal-weighting`, `solver:direct-scoring`, `complexity:polytime`, `maturity:standard`, `domain:information-retrieval`, `assumption:complete-rankings`, `assumption:strict-ranks`, `assumption:strict-positive-ranks`    
   h. **KemenyYoungAggregator** (*R-Kem*)    
      tags: `input:rank`, `family:consensus`, `mechanism:optimization`, `solver:exact-enumeration`, `solver:local-search`, `complexity:factorial-exact`, `complexity:heuristic-polytime`, `maturity:classical`, `domain:social-choice`, `assumption:complete-rankings`, `assumption:strict-ranks`

4. **Pairwise**.    
   Methods that derive the final ranking from pairwise comparisons, pairwise margins, or pairwise preference models.
   
   a. **Simple**.    
      Direct pairwise counting or margin-based aggregation rules.
      1. **CopelandPairwiseAggregator** (*R-Cop*)    
         tags: `input:pairwise`, `mechanism:counting`, `solver:direct-counting`, `complexity:polytime`, `maturity:classical`, `domain:social-choice`, `assumption:ties-native`
      2. **MarginRowSumAggregator** (*R-MRS*)    
         tags: `input:pairwise`, `mechanism:pairwise-margins`, `mechanism:scoring`, `solver:direct-scoring`, `complexity:polytime`, `maturity:standard`, `domain:sports-rating`, `assumption:ties-native`
   
   b. **Condorcet**.    
      Methods centered on majority comparisons and Condorcet-consistent reasoning.
      1. **MinimaxCondorcetAggregator** (*R-Minimax*)    
         tags: `input:pairwise`, `family:condorcet`, `mechanism:optimization`, `solver:direct-counting`, `complexity:polytime`, `maturity:classical`, `domain:social-choice`, `assumption:strict-ranks`
      2. **RankedPairsTidemanAggregator** (*R-RP*)    
         tags: `input:pairwise`, `family:condorcet`, `solver:greedy-dag`, `complexity:polytime`, `maturity:classical`, `domain:social-choice`, `assumption:strict-ranks`
      3. **SchulzeBeatpathAggregator** (*R-Sch*)    
         tags: `input:pairwise`, `family:condorcet`, `solver:path-closure`, `complexity:polytime`, `maturity:classical`, `domain:social-choice`, `assumption:ties-native`
      4. **SplitCycleAggregator** (*R-SC*)    
         tags: `input:pairwise`, `family:condorcet`, `solver:path-closure`, `complexity:polytime`, `maturity:specialized`, `domain:social-choice`, `assumption:ties-native`
      5. **RiverAggregator** (*R-River*)    
         tags: `input:pairwise`, `family:condorcet`, `solver:greedy-dag`, `complexity:polytime`, `maturity:specialized`, `domain:social-choice`, `assumption:ties-native`
      6. **StableVotingAggregator** (*R-SV*)    
         tags: `input:pairwise`, `family:condorcet`, `solver:recursive-elimination`, `complexity:exponential-recursive`, `maturity:specialized`, `domain:social-choice`, `assumption:ties-native`
      7. **SimpleStableVotingAggregator** (*R-SSV*)    
         tags: `input:pairwise`, `family:condorcet`, `solver:recursive-elimination`, `complexity:exponential-recursive`, `maturity:specialized`, `domain:social-choice`, `assumption:ties-native`
   
   c. **Probabilistic**.    
      Pairwise models that estimate latent preferences through a statistical likelihood model.
      1. **BradleyTerryAggregator** (*R-BT*)    
         tags: `input:pairwise`, `family:probabilistic`, `mechanism:likelihood`, `solver:mm-iteration`, `complexity:iterative-polytime`, `maturity:standard`, `domain:preference-learning`, `domain:sports-rating`, `assumption:ties-native`
      2. **ThurstoneMostellerAggregator** (*R-TM*)    
         tags: `input:pairwise`, `family:probabilistic`, `mechanism:likelihood`, `solver:gradient-ascent`, `complexity:iterative-polytime`, `maturity:standard`, `domain:psychometrics`, `domain:preference-learning`, `assumption:ties-native`
      3. **PolyRankAggregator** (*R-PolyRank*)    
         tags: `input:pairwise`, `family:probabilistic`, `mechanism:likelihood`, `solver:alternating-least-squares`, `solver:gaussian-elimination`, `complexity:iterative-polytime`, `maturity:specialized`, `domain:preference-learning`, `assumption:ties-native`
   
   d. **Stochastic**.    
      Methods that use random-walk or game-theoretic dynamics derived from pairwise preferences.
      1. **MarkovChainAggregator** (*R-MC*)    
         tags: `input:pairwise`, `mechanism:markov-chain`, `solver:power-iteration`, `complexity:iterative-polytime`, `maturity:standard`, `domain:preference-learning`, `assumption:strict-ranks`
      2. **MaximalLotteryAggregator** (*R-ML*)    
         tags: `input:pairwise`, `mechanism:game-theoretic`, `solver:fictitious-play`, `complexity:iterative-polytime`, `maturity:specialized`, `domain:social-choice`, `assumption:strict-ranks`
   
   e. **Linear**.    
      Methods that recover ratings by solving a linear system built from pairwise outcomes.
      1. **MasseyRankingAggregator** (*R-Massey*)    
         tags: `input:pairwise`, `mechanism:linear-system`, `solver:gaussian-elimination`, `complexity:polytime`, `maturity:standard`, `domain:sports-rating`, `assumption:strict-ranks`
      2. **ColleyRankingAggregator** (*R-Colley*)    
         tags: `input:pairwise`, `mechanism:linear-system`, `solver:gaussian-elimination`, `complexity:polytime`, `maturity:standard`, `domain:sports-rating`, `assumption:ties-native`
   
   f. **Optimization**
      Methods defined primarily through an optimization objective over pairwise preferences.
      1. **LinearOrderingProblemAggregator** (*R-LOP*)    
         tags: `input:pairwise`, `mechanism:optimization`, `solver:exact-enumeration`, `solver:local-search`, `complexity:factorial-exact`, `complexity:heuristic-polytime`, `maturity:specialized`, `domain:social-choice`, `assumption:ties-native`

5. **Probabilistic rank models**.    
   Methods that fit an explicit probability model over complete rankings or ranking generation.

   a. **PlackettLuceAggregator** (*R-PL*)    
      tags: `input:rank`, `family:probabilistic`, `mechanism:likelihood`, `solver:mm-iteration`, `complexity:iterative-polytime`, `maturity:standard`, `domain:preference-learning`, `assumption:complete-rankings`, `assumption:ties-broken`

6. **Multi-criteria decision analysis (MCDA)**.    
   Methods from decision analysis that combine multiple criteria, often with preference modeling or outranking logic.
   
   a. **PROMETHEEIIAggregator** (*R-P2*)    
      tags: `input:score`, `family:mcda`, `family:outranking`, `solver:flow-computation`, `complexity:polytime`, `maturity:standard`, `domain:mcda`, `assumption:ties-irrelevant`    
   b. **ELECTREIIIAggregator** (*R-E3*)    
      tags: `input:score`, `family:mcda`, `family:outranking`, `mechanism:thresholding`, `solver:flow-computation`, `complexity:polytime`, `maturity:standard`, `domain:mcda`, `assumption:ties-irrelevant`    
   c. **TOPSISAggregator** (*R-TOPSIS*)    
      tags: `input:score`, `family:mcda`, `mechanism:normalization`, `solver:distance-to-ideal`, `complexity:polytime`, `maturity:standard`, `domain:mcda`, `assumption:ties-irrelevant`    
   d. **VIKORAggregator** (*R-VIKOR*)    
      tags: `input:score`, `family:mcda`, `mechanism:normalization`, `solver:distance-to-ideal`, `complexity:polytime`, `maturity:standard`, `domain:mcda`, `assumption:ties-irrelevant`    

7. **Benchmark profiles**.    
   Methods designed around benchmark-wide comparison procedures such as statistical comparison protocols or performance profiles.
   
   a. **FriedmanNemenyiRankAggregator** (*R-N*)    
      tags: `input:rank`, `family:benchmark-profile`, `mechanism:statistical-test`, `solver:direct-statistic`, `complexity:polytime`, `maturity:standard`, `domain:benchmarking`, `assumption:complete-rankings`, `assumption:strict-ranks`    
   b. **DMAUCPerformanceProfileAggregator** (*DM-AUC*)    
      tags: `input:score`, `family:benchmark-profile`, `solver:profile-integration`, `complexity:polytime`, `maturity:standard`, `domain:benchmarking`, `assumption:ties-irrelevant`, `assumption:positive-scores`    
   c. **DMLBOLeaveOneOutProfileAggregator** (*DM-LBO*)    
      tags: `input:score`, `family:benchmark-profile`, `solver:profile-integration`, `complexity:polytime`, `maturity:specialized`, `domain:benchmarking`, `assumption:ties-irrelevant`, `assumption:positive-scores`    

### Curated tag vocabulary

Solver and complexity tags below describe the implementation used in this repository, not every possible implementation of the named method.

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

**Solver tags**

- `solver:direct-statistic`: computed directly from per-task values via a closed-form statistic
- `solver:direct-counting`: computed by direct counting of events, wins, or threshold hits
- `solver:direct-scoring`: computed by a direct additive scoring formula
- `solver:flow-computation`: computed from pairwise or outranking flows without iterative fitting
- `solver:path-closure`: computed from strongest-path or path-closure style graph updates
- `solver:greedy-dag`: computed by greedily building an acyclic precedence graph
- `solver:recursive-elimination`: computed by recursive or memoized winner elimination
- `solver:mm-iteration`: fitted by minorization-maximization style fixed-point updates
- `solver:gradient-ascent`: fitted by gradient-based optimization
- `solver:alternating-least-squares`: alternates between coefficient fitting and score estimation
- `solver:power-iteration`: computed from the stationary distribution of a transition matrix
- `solver:fictitious-play`: approximated by deterministic fictitious play
- `solver:gaussian-elimination`: solved as a linear system with Gaussian elimination
- `solver:exact-enumeration`: solved exactly by enumerating all permutations
- `solver:local-search`: approximated by deterministic swap-based local search
- `solver:distance-to-ideal`: computed from distances to ideal and anti-ideal reference points
- `solver:profile-integration`: computed by integrating or repeatedly recomputing benchmark profiles

**Complexity tags**

- `complexity:polytime`: implemented with polynomial-time deterministic computation
- `complexity:iterative-polytime`: implemented by an iterative procedure with polynomial-time iterations
- `complexity:factorial-exact`: exact implementation based on permutation enumeration
- `complexity:exponential-recursive`: recursive subset-based implementation with exponential worst-case growth
- `complexity:heuristic-polytime`: heuristic implementation with polynomial-time iterations

**Maturity tags**

- `maturity:classical`: long-established canonical method
- `maturity:standard`: broadly recognized and commonly used baseline
- `maturity:specialized`: mostly niche, research-oriented, or community-specific

**Domain tags**

- `domain:benchmarking`: used primarily for benchmark aggregation and empirical comparison studies
- `domain:social-choice`: used primarily in voting, preference aggregation, or social choice
- `domain:information-retrieval`: used primarily in retrieval, search, or rank fusion
- `domain:sports-rating`: used primarily for competitive rating and schedule-based ranking
- `domain:mcda`: used primarily in multi-criteria decision analysis
- `domain:preference-learning`: used primarily in probabilistic preference or ranking models
- `domain:psychometrics`: used primarily in latent-trait or comparative judgment modeling

**Assumption tags**

- `assumption:complete-rankings`: expects one rank per algorithm for every task
- `assumption:ties-irrelevant`: tie structure is not part of the method's input model or decision rule
- `assumption:ties-native`: tied inputs are explicitly represented in the implementation rather than silently broken
- `assumption:ties-broken`: tied inputs are accepted but resolved by deterministic tie-breaking in the implementation
- `assumption:strict-ranks`: the method is best interpreted on strict rankings; tied ranks should be avoided
- `assumption:strict-positive-ranks`: requires strictly positive rank positions
- `assumption:nonnegative-scores`: requires non-negative score values
- `assumption:positive-scores`: requires strictly positive score values

---


### Methods found somewhere, but not implemented as separate classes

The following methods are intentionally not exposed as standalone implementations:

- `Plurality Vote`
  - Conceptually equivalent to `R-B` (`BestRankCountAggregator`), but under a different name.
- `Normalized Score Aggregation`
  - Close in spirit to `Q-RM` (`RescaledMeanQualityAggregator`), but not identical:
    `Normalized Score Aggregation` uses `L_t/H_t` scaling, while `Q-RM` uses per-task min-max scaling over the observed algorithms.
