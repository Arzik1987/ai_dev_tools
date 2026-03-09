A good final package structure could be:


```text
score_based
rank_based
pairwise
  simple
  condorcet
  probabilistic_pairwise
  stochastic_game_theoretic
  linear_system
  optimization
probabilistic_rank_models
mcda_outranking
benchmark_profiles
```

```text
score_based:
  Q-M, Q-Md, Q-GM, Q-HM, Q-RM, Q-Th(theta)

rank_based:
  R-M, R-Md, R-B, R-W, R-Borda, R-Dowdall, R-RRF, R-Kem

pairwise/simple:
  R-Cop, R-MRS

pairwise/condorcet:
  R-Minimax, R-RP, R-Sch, R-SC, R-River, R-SV, R-SSV

pairwise/probabilistic:
  R-BT, R-TM, R-PolyRank

pairwise/stochastic:
  R-MC, R-ML

pairwise/linear:
  R-Massey, R-Colley

pairwise/optimization:
  R-LOP

probabilistic_rank_models:
  R-PL

mcda:
  R-P2, R-E3, R-TOPSIS, R-VIKOR

benchmark_profiles:
  R-N, DM-AUC, DM-LBO
```

