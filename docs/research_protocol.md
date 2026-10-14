# Shared evidence forecasting: research protocol v0

Status: proposed design v0, 2026-10-14. No main evaluation has been run. This document fixes the scientific questions and target grid before the main evaluation. It is not an external preregistration. Implementation checks and an independent precision pilot must pass before calling a run confirmatory. Any change after inspecting outcomes must be recorded, dated, and labelled exploratory. No human or AI forecasts are measured by this protocol.

## Question and contribution

How much does a learned scalar correction improve aggregation of individually calibrated forecasts when their evidence overlaps? How many resolved questions does it need, and where does a scalar correction fail?

The contribution is a reproducible explanatory experiment with an exact reference posterior, finite-training uncertainty, and a structured-overlap failure case. It is not a new theory of forecast aggregation. Overlap-sensitive extremizing and learned logit aggregation already have substantial prior literature. The amount of data sufficient for this simulator is not a sample-size prescription for actual forecasting teams.

## Primary literature and relation to this project

- Satopää, Pemantle, and Ungar, *Modeling Probability Forecasts via Information Diversity*, [author preprint, arXiv:1406.2148](https://arxiv.org/abs/1406.2148), and [author-hosted final article](https://www2.math.upenn.edu/~pemantle/papers/aggregation.pdf). The partial-information framework explicitly connects overlap to extremizing. Our binary Gaussian likelihood model is an elementary illustration of related principles, not a reproduction of their Gaussian-process/probit construction. The alternate arXiv submission 1501.06943 was withdrawn as a duplicate, so it is not a separate reference or evidence against the work.
- Satopää, Baron, Foster, Mellers, Tetlock, and Ungar (2014), *Combining multiple probability predictions using a simple logit model*, International Journal of Forecasting 30(2), 344–356, [publisher record and abstract](https://doi.org/10.1016/j.ijforecast.2013.09.009). This provides direct precedent for a simple learned logit aggregator. Our fitted scalar should be described as a baseline family with known precedent.
- Babichenko and Garber (2018), *Learning of Optimal Forecast Aggregation in Partial Evidence Environments*, [author preprint](https://arxiv.org/abs/1802.07107). The paper studies learning from forecasts and resolved outcomes under partial evidence, including possible and impossible cases. Its information-structure assumptions differ from this Gaussian simulator; its results do not establish identification for arbitrary real forecasters.
- Gneiting and Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 102(477), 359–378, [author-hosted paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). Proper scoring rules motivate Brier and logarithmic losses as evaluation targets. We use the loss convention, where smaller is better.

These references establish precedent and evaluation principles. The derivations below follow directly from the stated data-generating model.

## Exact exchangeable model

For each independent question, draw Y ~ Bernoulli(1/2), Z = 2Y - 1, and independent standard normal U, E_1, ..., E_n. Let

    S_i = a Z + sqrt(rho) U + sqrt(1-rho) E_i,
    a > 0, 0 <= rho <= 1.

Conditional on Y, each S_i has variance one and mean aZ. Its log likelihood ratio is

    ell_i = log[f(S_i | Y=1) / f(S_i | Y=0)] = 2a S_i.

Thus p_i = logistic(ell_i) is the exact individual posterior, and is calibrated in the population. This is stronger than calibration alone: calibration of arbitrary reports does not imply any of the following joint formulas.

For rho < 1, the conditional covariance is

    Sigma = (1-rho) I + rho 11',
    Sigma 1 = [1+(n-1)rho] 1.

Expanding the two multivariate Gaussian quadratic forms gives

    L* = log[f(S | Y=1) / f(S | Y=0)]
       = 2a 1' Sigma^(-1) S
       = sum_i ell_i / [1+(n-1)rho].

The exact aggregate is logistic(L*). At rho=1, all model-generated signals coincide. The nonsingular expression extends continuously to their common log odds; arbitrary disagreeing inputs at rho=1 are outside the model support.

Define n_eff = n/[1+(n-1)rho]. Then L* = n_eff mean_i ell_i. This is an information-equivalent signal count in this model, not the number of independent people in a population. At rho=0 all likelihood evidence adds. At rho=1 it is counted once. For fixed positive rho and increasing n, n_eff approaches 1/rho.

For an independent implementation check, write d = a^2 n_eff. Conditional on Y=1, L* is normal with mean 2d and variance 4d; conditional on Y=0 its mean is -2d with the same variance. This gives a one-dimensional route to reference scores and checks the vector simulator.

With a known non-half common prior pi and identical Gaussian likelihoods, b0 = logit(pi), ell_i = b0 + 2aS_i, and

    L* = b0 + sum_i (ell_i-b0)/[1+(n-1)rho].

The v0 core uses pi=1/2. Simply adding posterior logits with another prior counts that prior repeatedly.

## Correlation and identification

Rho is correlation of the signals conditional on the outcome. For the half prior,

    Cov(S_i,S_j) = a^2 + rho,  i != j,
    Var(S_i) = a^2 + 1,
    Corr(S_i,S_j) = (a^2+rho)/(a^2+1).

The same marginal correlation holds for the logits because they share a positive linear scaling. It does not generally hold for the probabilities after the nonlinear logistic transformation. Strong forecasters can agree because they track the same outcome even with conditionally independent noise. Raw agreement or raw forecast correlation is therefore not rho.

Inside the correctly specified model, labelled repeated questions identify the population scalar beta*=1/[1+(n-1)rho] in logistic regression on X=sum_i ell_i. Mapping a fitted beta to rho is justified only by this model and n>1. Across general data-generating processes, a fitted slope can reflect report miscalibration, varying strength, omitted priors, or distribution shift. It does not identify shared sources or a causal overlap mechanism. One question's forecast vector cannot determine overlap without structural assumptions.

A learned zero-intercept nonnegative scalar cannot reverse the sign of X; a zero scalar returns the prior. When unequal source groups cause majority double counting to reverse that sign relative to the group oracle, rescaling confidence alone cannot fix the decision direction.

## Source-group extension

Assign every forecaster to a fixed, known group g. Draw independent group factors U_g and private E_i, and set

    S_i = aZ + sqrt(rho) U_g(i) + sqrt(1-rho) E_i.

Within-group conditional correlation is rho; between-group conditional correlation is zero. Conditional covariance is block diagonal, each group g of size m_g having block (1-rho)I + rho11'. Therefore

    L*_group = sum_g [sum_(i in g) ell_i / (1+(m_g-1)rho)].

At rho=1, identical reports within each group count once. At rho=0, all reports count separately. Unequal groups need unequal weights; replacing rho with an average pairwise correlation generally does not recover the exact posterior. If all groups have equal size, one scalar can recover this oracle. Group labels and true rho are privileged oracle information, not learned evidence.

A worked illustration is nine identical p=.6 reports from one source and one independent p=.1 report. Counting ten reports independently gives logistic(9 logit(.6)+logit(.1)), about .81. Counting the two independent sources once gives logistic(logit(.6)+logit(.1))=1/7. This is a constructed support-consistent limiting example, not an empirical frequency or a performance estimate.

## Target grid and planned comparisons

The exchangeable target grid is the Cartesian product:

| Variable | Values |
| --- | --- |
| Forecasters n | 2, 5, 10, 20 |
| Conditional overlap rho | 0, .1, .25, .5, .75, 1 |
| Signal strength a | .5, 1.0 |
| Training questions M | 50, 200, 1000, 5000 |
| Prior | .5 |

The primary condition is n=10, rho=.5, a=.5, M=200. Other sample sizes give the learning curve; other model cells assess sensitivity. Do not select the headline cell by observed improvement. The reference beta in this primary condition is 1/5.5 and n_eff=10/5.5.

Required methods are arithmetic mean of probabilities; independent-evidence pool logistic(sum ell_i); scalar learned pool logistic(beta_hat sum ell_i); and exact oracle. The current implementation also includes the prior and a fixed first forecaster as diagnostics. The geometric odds pool logistic(mean ell_i) is an optional comparison pending implementation. Never select the best test-set forecaster.

Fit beta on training log loss with zero intercept and beta constrained to [0,1], matching `fit_discount` in `shared_evidence/evaluate.py`. This interval contains every exchangeable oracle under the model and permits shrinkage all the way to the prior. Values below 1/n are allowed as finite-sample fits but do not correspond to rho in [0,1] under the exchangeable model. Use the deterministic convex gradient-bisection optimizer with 80 iterations, include endpoints, and record boundary hits and convergence. A training set with all summed logits equal to zero returns beta=0. Do not select fitting loss, bounds, or stopping tolerance after seeing test performance. A free slope or intercept is a separately labelled sensitivity experiment, not a quiet change to this learner.

The source-group stress cases target n=10, a=.5, rho=.9, partitions [5,5], [9,1], and [4,3,2,1], using the same training sizes. These are secondary diagnostics. If the runner implements only a subset, report that subset explicitly and do not present the missing cells as executed.

## Sampling, freeze, and leakage

Before the production run, record a machine-readable manifest with the exact grid, method names, optimizer settings, master seed, seed-derivation algorithm, training and test counts, repetitions, code hashes, and protocol hash. Keep pilot and production streams distinct. A pilot may determine runtime and required precision; it may not select favorable effects or remove unfavorable cells. This v0 becomes a frozen executed protocol only when the manifest and implementation agree.

Use independent questions as sampling units. All methods in a cell must receive the same test questions for paired comparisons. Training, test, and pilot question draws must have disjoint random streams. Never reuse a test question as training for a different method in the same reported comparison. Training sizes may use nested prefixes within a repetition if this is recorded; resulting learning-curve points are dependent.

The proposed precision target for the primary condition is 30 independent training/test repetitions with 10,000 test questions per repetition; these counts become binding when the production manifest is frozen. Secondary sensitivity cells may use smaller fixed replication and test counts chosen from development precision/runtime checks and recorded before production. Such cells remain descriptive and must report their actual counts. A smaller development run is a pilot; a separately prespecified smaller production sensitivity cell is not a primary evaluation. Runtime or precision changes require a dated manifest amendment before the main evaluation. Do not stop early when an effect reaches significance.

The current development pilot configuration, `configs/development-pilot-v0.json`, uses phase namespace `development-pilot-v0`, seed 20261014, 3 repetitions, 200 training questions and 2,000 test questions per repetition, and n=10, a=.5 in all four scenarios: exchangeable rho=0, .5, and 1, plus a nine-plus-one source partition at within-group rho=1 and between-group rho=0. These are development checks, not the main evaluation or evidence that the full target grid has run. A production run must use a distinct phase namespace and seed. The production uncertainty and precision gates below remain pending.

A second development configuration, `configs/development-groups-v0.json`, uses phase namespace `development-groups-v0`, seed 20261015, the same 3 by 200/2,000 sample counts, and group partitions [5,5], [9,1], and [4,3,2,1] at n=10, a=.5, within-group rho=.9, and between-group rho=0. It is also diagnostic, with no confirmatory interpretation.

Fit only on training logits and outcomes. Ground-truth rho, a, noise draws, and source labels must not reach the learned scalar except that n is observable. The oracle deliberately receives generative parameters and is a correctness reference. Never choose a method, source partition, calibration binning, or display range using test outcomes. Record failed fits rather than deleting them; a failed fit invalidates that cell's learned-method result until the cause is resolved and the rerun is documented.

For a future real dataset, split by underlying resolved event, time, and source provenance as appropriate. Multiple timestamps or paraphrases of the same event cannot cross the split. Freeze inclusion criteria before evaluating outcomes, record unresolved events and missing reports, and audit any model's potential access to resolutions. This synthetic experiment has no survivorship or training-corpus leakage claim about real data.

## Metrics and uncertainty

Primary outcome: paired mean held-out log loss difference, learned minus comparator, in the primary condition at M=200. Comparators are arithmetic mean and independence; negative is better. Log loss uses natural logarithms and is measured in nats per question. Report both contrasts, absolute scores, uncertainty, and the number of independent repetitions. To claim improvement over both, both contrasts must support improvement. Use Bonferroni-adjusted 97.5% two-sided intervals for the two primary contrasts; ordinary 95% intervals elsewhere are descriptive.

Secondary outcomes: Brier loss, learned-minus-oracle regret, learning curves, beta estimates, boundary-hit frequency, and group stress effects. Brier loss is (p-Y)^2 with no factor of two. Geometric-pool comparisons remain optional pending implementation. Compute log loss directly from a finite logit L as max(L,0)-YL+log1p(exp(-abs(L))) to avoid arbitrary clipping and overflow. If a probability-only method requires clipping, record the threshold and clipping count; do not conceal extreme-error penalties.

Pre-production amendment, 2026-10-14: following a separate 30-repetition by 10,000-test-question development precision pilot in the primary condition, replace the previously proposed Student-t interval with a prespecified percentile bootstrap over independent repetition means. The reason is a reproducible uncertainty implementation using the standard library; the method must not be selected according to which interval gives a preferred significance result. Pilot observations remain excluded from production, and this amendment does not claim that the main evaluation has run.

For each repetition r, compute its mean paired question-loss difference D_r. For each of 10,000 bootstrap resamples, draw R repetition indices with replacement and average the corresponding D_r values. Resample whole paired repetition summaries, using the same index draw for both primary contrasts. The primary intervals use the 1.25th and 98.75th percentiles of these bootstrap means (nominal 97.5% two-sided intervals); descriptive 95% intervals use the 2.5th and 97.5th percentiles. Fix the bootstrap seed in the production manifest, independently of simulation streams. For sorted B bootstrap estimates v, define percentile q by linear interpolation at zero-based index (B-1)q. Report the original mean D, the sample standard deviation across repetitions, R, B, and both interval endpoints. Do not bootstrap forecasters or treat questions sharing a fitted beta as independent training repetitions.

This resampling unit captures variation from fresh training and fresh test sets. Percentile coverage is approximate: 30 repetitions can poorly represent skewed or heavy-tailed training effects, and 10,000 resamples reduce Monte Carlo quantile noise without increasing the number of independent fits. Bonferroni adjustment controls familywise error only to the extent that each interval attains its nominal coverage; it does not cure undercoverage of the base method. Compared with a Student-t interval, the percentile interval can be asymmetric but does not studentize or correct bias, and is not automatically more accurate. See Efron (1987), [Better Bootstrap Confidence Intervals](https://doi.org/10.1080/01621459.1987.10478410), for the distinction between percentile and improved bootstrap methods. Preserve the repetition summaries so these limitations can be inspected. Do not claim exact finite-sample coverage or silently switch interval methods after production. Smaller sensitivity samples have weaker uncertainty estimates and remain descriptive.

Do not treat the n reports per question as independent outcome observations. If only one training fit is evaluated, label its paired question interval conditional on that fit; it does not quantify training uncertainty. Reused test sets across repetitions require a different variance analysis and must not be treated as independent repetitions.

Reliability diagrams use ten fixed equal-width bins on [0,1], with counts and event-frequency uncertainty shown. They are diagnostics, not a third primary endpoint; sparse bins and binning sensitivity must be visible. Do not equate a small binned calibration error with optimal prediction. Skill rankings can differ between Brier and log loss; report both without choosing the more favorable one.

## Falsification and interpretation

The Gaussian posterior derivation is an algebraic claim. If the oracle fails direct density-ratio checks, limiting cases, or independent moment checks, stop and repair the implementation. A finite test sample may occasionally score another method better than the oracle; proper scoring optimality is a population statement, not a deterministic ordering on every sample.

The empirical claim under test is that the specified learned scalar improves the specified baselines at a modest training size in the primary simulator condition. An interval including zero is inconclusive at the selected precision; worse performance is a negative result. Either outcome must be retained. Do not infer success from oracle superiority alone.

Failure on unequal source groups shows a structural limit of scalar confidence adjustment. Success under exchangeability establishes only that one parameter can be learned in a one-parameter family. All reports are generated from a common correct likelihood model; they have equal strength, a common prior, no strategic distortion, no missingness, and no changing information environment. Real humans and language models need not satisfy these assumptions. Gaussian dependence is a mathematical stand-in for redundancy and is not a literal model of copying text or citing the same article.

## Validation and release gates

1. Derive the posterior independently from Gaussian densities and compare it numerically with the closed form. Check permutation invariance, n=1, rho=0, rho=1, outcome-label symmetry, and finite extreme logits.
2. Verify individual conditional means/variances, covariance, and the one-dimensional oracle-logit moments with stochastic tolerances fixed before running checks.
3. Validate the group formula against a separate linear-system or density calculation and verify equal-size and fully repeated-source limits.
4. Test fitted beta on an independently generated large training set against the known population beta with a justified tolerance, plus endpoint, invalid-input, and deterministic-replay checks.
5. Verify seed separation, paired method evaluation, stable log loss, manifest serialization, optimizer diagnostics, and preservation of all grid cells. Numerical unit tests must not merely restate implementation expressions.
6. Run the precision pilot, archive its role and actual counts, freeze the production manifest, then execute the fixed evaluation once. Report all failures and deviations.
7. Regenerate article tables and figures from saved machine-readable results. Link the manifest and actual validation report. Every reported number must have a traceable result field; do not hand-copy rounded values into an unverifiable narrative.

Passing code tests establishes implementation checks. Passing the synthetic evaluation supports claims about the declared simulator. Neither establishes usefulness on real forecasting teams, commercial value, causal source identification, or improved AI reasoning.
