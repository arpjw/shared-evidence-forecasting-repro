# Learning source groups from resolved forecasts

Design proposal, 2026-10-14. This is a fresh synthetic follow-up designed after
observing the primary and stress-v1 results. It is not an external preregistration
or a replication of the original primary claim. This file becomes binding only
when an implementation and configuration manifest hashes it before any final
holdout is generated or inspected. Earlier files and results remain unchanged.

## Question

Can a learner recover useful conditional dependence from repeated resolved
forecasts, without source labels, and use it to repair the scalar learner's
nine-plus-one failure? The claim is limited to stable independent Gaussian
source groups, equal signal strength and a half prior. Improving prediction is
distinct from identifying real provenance.

Satopää, Pemantle and Ungar's [information diversity model](https://arxiv.org/abs/1406.2148)
provides precedent for relating overlap to probability aggregation.
Babichenko and Garber's [partial evidence learning paper](https://arxiv.org/abs/1802.07107)
studies aggregation with unknown information structure and shows that learning
possibilities depend on assumptions. Our elementary conditional-correlation
estimator is an explanatory synthetic experiment, not a claimed new general
identification theorem or a reproduction of either model.

## Fixed learner and information boundary

The learner receives only forecast logits and resolved binary outcomes from
training. It never receives true groups, rho, signal strength, simulator signals,
noise draws or test outcomes. Columns identify the same forecasters throughout
a repetition; stable identity is a substantive assumption. Convert simulator
Question objects to ForecastRecord before fitting. The oracle alone receives
true groups and rho.

Set Z=2Y-1 and X_i=Z*ell_i for each training question. Under the specified
symmetric Gaussian simulator, E[X_i]=2a^2 and Cov(X)=4a^2 Sigma. Center each X
column using its training mean and compute Pearson correlations from sums of
centered products. A zero-variance column has all recorded correlations zero,
including its diagonal; count those columns from the zero diagonal entries.
This transformation depends
on resolved outcomes: the learner is source-label-free, not unsupervised.

Join pairs whose estimated correlation strictly exceeds

    t = min(1, sqrt(2 log(n(n-1)/0.05)/(M-1))).

Require M>=3; for n=1 use threshold 1. Also join exact duplicate original
training-logit columns when their signed residual variance is positive. This
handles the fully duplicated source limit even when the threshold reaches one. This fixed threshold is a conservative
heuristic, not a proved familywise significance bound for dependent sample
correlations. Connected components define groups, ordered by smallest column
index. Estimate common rho as the mean of all within-component pair
correlations, clipped to [0,1]; with no pairs use rho=0. Give forecaster i in a
group of size k weight 1/[1+(k-1)rho_hat]. Fit a final zero-intercept scalar
alpha in [0,1] to weighted training logit sums using the existing 80-step
bisection optimizer. The final prediction is alpha times the weighted sum.
Retain threshold, correlation matrix, partition, rho, weights and alpha.

The scalar comparator fits its original [0,1] slope on the same M training
questions. Required test methods are prior, arithmetic probability mean,
independent evidence, scalar learner, inferred groups and true group oracle.
There is no validation split because there is no hyperparameter or candidate
selection. All algorithm settings are fixed before the fresh evaluation.
The group scalar only reduces confidence; it cannot compensate for excessive
estimated overlap by increasing confidence above the estimated group pool.
Connected components can merge groups through a single noisy edge. Both are
explicit finite-sample failure modes to retain.

## Fresh splits and fixed evaluation

All conditions use n=10, a=.5, prior=.5 and the existing Gaussian group simulator.
The primary follow-up cell is partition [9,1], rho=.9, 400 training questions, 30 independent repetitions and 5,000 test questions per
repetition. This is a new 400-resolved-question design, not a claim to match the
old 200-question training budget.

The descriptive sensitivity grid is partitions [9,1], [5,5] and [4,3,2,1], rho
in {0,.5,.9,1}, training counts {100,400,2000},
12 repetitions and 2,000 test questions per repetition. It has 36 cells and
864,000 test questions. Training-size streams are independent, not nested.
The separate primary cell adds 150,000 test questions. Run every declared cell;
never select favorable conditions for the report.

The development pilot uses seed 95174200 and phase source-inference-development-v1.
The primary uses seed 95174300 and phase source-inference-primary-v1. Sensitivity
configs use seeds 95174400, 95174401 and 95174402 in training-count order, with
phase source-inference-sensitivity-m{M}-v1. Use derive_seed with config seed,
phase, condition name, repetition and split name (train/test).
This separates conditions, repetitions, splits, old experiments and the pilot.
Use root bootstrap seed 95174999, deriving each cell's stream with
derive_seed(root, phase, condition name, "paired-bootstrap"). Use 10,000 resamples and linear
interpolated percentiles at index (B-1)q. Same bootstrap repetition indices
must serve all comparisons within a cell.

A development run may assess runtime and code behavior. Pilot observations are
excluded from final inference. Any count or algorithm change before the final
freeze must be documented here; after final outcomes are inspected it requires
a new protocol and holdout. No early stopping or follow-up seed search is allowed.

## Metrics and predeclared scientific gates

Primary metric is mean test log loss in nats. Compute differences within each
independent repetition, then bootstrap whole paired repetition summaries.
Two simultaneous primary statements use 97.5% two-sided percentile intervals:

1. Inferred-group loss minus scalar loss has upper endpoint below 0.
2. Inferred-group loss minus oracle loss has upper endpoint below 0.01 nats.

The second gate is a prespecified practical proximity threshold, not evidence
of equality or proof of exact oracle recovery. The joint scientific claim
requires both gates. A failure or inconclusive interval remains a valid result.
Intervals have approximate finite-repetition coverage; adjustment cannot repair
bootstrap undercoverage. Sensitivity intervals are descriptive 95%, with no
multiplicity-adjusted discovery claim.

Report every method's Brier and log loss, the two primary contrasts, absolute
regrets, replicate variability, and all fits. Report the mean gap fraction
closed as 1 - mean(inferred minus oracle)/mean(scalar minus oracle) only where
the denominator is positive, flag denominators below .001 nats as unstable,
and always show the underlying regrets. This descriptive ratio is not a gate.

Group diagnostics are inferred group count, rho estimate, exact partition
recovery up to label renaming, and pairwise co-membership error across the 45
pairs. At rho=0 or a=0, true partition labels are observationally unidentifiable;
mark group-recovery diagnostics not applicable. At a=0 all logits vanish and aggregation is the prior. A shared unconditional outcome
signal is not evidence of a source group. Compare strict opposite signs between
inferred and oracle logits, and scalar and oracle logits; count exact zero ties
separately. Report the fraction of scalar/oracle opposite-sign questions where
the inferred learner agrees with the oracle. Oracle disagreement measures a
reference decision discrepancy, not realized classification accuracy.

## Validation and completion

Before freezing, test signed centering, zero-variance behavior, graph transitivity, deterministic thresholds, partition label invariance,
column permutations, true-rho endpoints, equal group sizes and finite extreme
logits. Validate that fitting cannot access privileged metadata. Check a hand
constructed example where outcome agreement creates raw correlation while
conditional noise is independent. Check that changing held-out outcomes cannot
change fitted state. Hash protocol, configs, sources and Python version.

Save all generated rows, split identities, fitted states,
test predictions and per-repetition summaries in a compressed ledger or an
equivalently replayable archive. Independently regenerate every question and
recompute every fit, prediction and score. Checksums alone
are insufficient. An independent verifier must also check method information
boundaries and stream separation. Record source hashes before and after runs.
Runtime/storage failures require an explicit dated amendment and complete rerun
of affected cells; numerical bugs require a regression check and new source
binding. Preserve failed receipts and explain invalidated results.

Completion requires a full validated grid and a larger article that includes
both successes and failures. It never requires a favorable scientific gate.
The final article must distinguish the original primary result, exploratory
stress results and this fresh synthetic follow-up. Stable forecaster identity,
fixed groups, equal calibration and strength, a shared half prior, Gaussian
noise, and no missing forecasts are essential limits. The experiment provides
no causal identification of real sources, no human or AI validation, no
performance under distribution shift, and no market-trading evidence.
