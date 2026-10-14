# Synthetic stress protocol v1

Dated 2026-10-14. This protocol was specified after inspecting the completed
primary evaluation. Every stress result is exploratory, even when these files
are hashed before execution. The original primary config, manifest, ledger and
reports remain the evidence for the original primary claim. This document
extends `docs/research_protocol.md`; it does not amend that frozen experiment.

## Purpose and completion rule

Measure sensitivity to strength, overlap, ensemble size, finite training, and
unequal source groups. A successful research workflow produces a complete,
replayable account of its favorable and unfavorable results. It does not require
that the scalar learner beat every baseline. A structural failure of a scalar
learner is a result to retain. An implementation or validation failure triggers
repair followed by a complete rerun of every affected cell and its regression
checks. Never alter a cell, seed, bound, or scoring rule to obtain improvement.

All experiments use local standard-library Python and synthetic questions. No
human forecasts, AI forecasts, market data, transactions, or paid resources are
involved. All models retain prior 0.5, equal signal strength within a cell, unit
conditional marginal variance, and Gaussian noise.

## Fixed grid and random streams

| Family | Configs | Conditions per config | Independent fits per condition | Train counts | Test questions per fit |
| --- | --- | --- | --- | --- | --- |
| Exchangeable | `stress-exchangeable-m{M}-v1.json` | 48 | 8 | 50, 200, 1000 | 1000 |
| Source groups | `stress-groups-m{M}-v1.json` | 12 | 12 | 50, 200, 1000, 5000 | 2000 |

Exchangeable conditions are the full product n in {2,5,10,20}, rho in
{0,.1,.25,.5,.75,1}, a in {.5,1}. This executes 144 of the 192 exchangeable
cells proposed in the original protocol. The M=5000 exchangeable slice is not
included; do not imply that it ran. Source-group conditions are the full product
of partitions [5,5], [9,1], [4,3,2,1] and within-group rho in {0,.5,.9,1},
at n=10 and a=.5. Between-group correlation is zero. This executes 48 group
cells, including all originally proposed rho=.9 training sizes.

Total: 192 condition/training-size cells, 1,728 independent fitted models,
2,304,000 held-out question rows and 1,380,000 training question rows. Question
rows across conditions are not all independent, as explained below.

Exchangeable seeds, in increasing M order, are 84173000, 84173001, 84173002.
Group seeds are 84173100 through 84173103. Each config has a distinct phase
namespace `exploratory-stress-{family}-m{M}-v1`. The existing SHA256 seed
construction and phase/train/test/replicate namespaces remain in force.
Bootstrap seeds follow `derive_seed(config.seed, 'replicate-analysis-v1',
config.phase)`. Use 10,000 bootstrap resamples per reported interval.

The current simulator shares outcomes and private draws across conditions in
a config through common random numbers. Replicates within a condition have
separate streams. Training sizes use distinct seeds and phases, not nested
prefixes. Keep within-condition pairing across methods. Do not pool condition
means as independent observations or interpret aggregate cell counts as a
sample size for statistical inference. Oracle and learner receive the same
held-out questions.

## Fixed methods and outputs

Retain all existing methods: prior, first forecaster, arithmetic mean,
independent pool, fitted scalar and the appropriate oracle. In group cells,
retain the exchangeable formula under that name, never as the correct oracle.
Fit the scalar on training outcomes and summed logits only, with the original
zero intercept, [0,1] bounds and deterministic 80-step bisection. Never give it
source labels or true overlap. Preserve every replicate and boundary fit.

For each cell report log loss and Brier scores for all methods, learned minus
arithmetic mean, learned minus independence, learned minus oracle, all fitted
slopes, slope boundary frequency, and means with replicate variability. Use
95% descriptive percentile intervals resampling entire fitted replicates. If a
shared analysis entry point emits its default 97.5% intervals, report their
actual level explicitly; that output is still descriptive. Neither interval
level controls multiplicity over this grid. Do not classify cells as discoveries
by scanning for intervals excluding zero. Eight and twelve independent fits
provide limited tail precision. Publish the replicate rows to show that limit.

For group cells, add the fraction of held-out questions where the sign of the
summed report logits differs from the sign of the group-oracle logit. Use
strict opposite signs; exact zero counts separately. A nonnegative scalar
cannot reverse the sign of a nonzero sum. At beta=0 it can replace the decision
with a tie, without recovering the oracle direction. For unequal group sizes,
this diagnostic isolates information lost by using one total sum even with
unlimited training. Record mean absolute oracle probability distance from .5
on opposite-sign questions when available, to distinguish tiny disagreements
from consequential ones. This diagnostic uses privileged labels and overlap
only for evaluation.

Optional figures use the full fixed grid with consistent axes: learned-minus-
arithmetic log loss by n/rho/a/M, learned-minus-oracle loss by group/rho/M,
and fitted slope distributions. All plotted values must come from archived
machine-readable results. Any omitted optional diagnostic must be named in the
report; none can silently be described as completed.

## Validation, failures and repair loop

Before executing the suite, save a separate suite manifest containing this
protocol hash, all seven config hashes, source hashes, Python version and exact
analysis settings. The existing primary freeze helper hardcodes a primary
scope and requires a production namespace. Do not relabel these exploratory
configs to make that helper accept them. A dedicated exploratory suite receipt
may bind the same sources without pretending this is another primary run.
Freeze before inspecting stress outcomes. Preserve its timestamp and hash.

Implementation gates are: valid complete config grid; deterministic streams;
finite scores; no duplicate or missing cells; matching source hashes throughout
each run; independent oracle checks; stable extreme-logit loss; and fit
optimality/boundary checks. Also verify that every reported method is paired
on the same held-out questions and that train/test streams differ. Replay all
saved summaries from config and source, or independently replay a complete
compressed ledger. A checksum alone establishes file identity, not correctness.
A summary-only replay cannot be advertised as per-question ledger verification.

Critical algebraic controls are rho=0 (group oracle and independence coincide),
equal group sizes (one scalar can represent the group oracle), and rho=1
(reports within a group duplicate a single signal). Check single-forecaster,
permutation, label-symmetry, zero-strength limit, near-endpoint overlap and
extreme-logit inputs in implementation tests as appropriate. These additional
unit-level controls are not extra empirical cells in the fixed grid.

At a gate failure, retain the failing input and diagnostic; assign an agent to
reproduce and repair it, and another review path to validate the fix. Add a
regression check that fails on the reproduced defect. Hash the corrected code
in a versioned amendment, explain which old results became invalid, and rerun
all affected cells plus required controls. Never overwrite the primary archive
or erase the first failure. Results from changed code must carry the new source
hash. Repeat until implementation gates pass and the complete grid is valid.
If a validation problem cannot be resolved, report the affected results as
invalid and the precise blocker rather than claiming success.

Finite-sample score orderings are not implementation gates. An oracle can lose
on a finite test batch. A scalar losing to arithmetic mean in an independent
or redundant limit is also admissible. Investigate surprising effects using
independent density and simulation checks; preserve the observed loss even if
it conflicts with the motivating story. If a new method is developed after
seeing these outcomes, freeze and evaluate it on a fresh, explicitly separate
holdout. Do not turn the current stress set into both model selection and final
evidence. Improving numerical correctness does not justify rewriting the
scientific claim to universal scalar superiority.

## Interpretation boundaries

The primary result estimates performance in one specified exchangeable
Gaussian cell. This suite maps a broader but still narrow synthetic family.
Learning curves distinguish finite training variability from a persistent gap
to the known group oracle. At equal group sizes, diminishing regret is expected;
at unequal sizes, a gap and opposite-sign decisions are plausible structural
failure mechanisms. These are questions tested by the suite, not conditions
for accepting its result.

No distribution shift, unequal marginal calibration, source-label error,
missing forecasts, strategic reports, non-half prior, time dependence, or real
forecasting team is tested here. A learned slope need not identify overlap
outside this model. There are no transaction-cost or survivorship assumptions
to validate because this is not a trading backtest. The synthetic split controls
its own training/test leakage; it establishes nothing about model training
corpora or real event leakage. The final article must include the observed
failure cases, limited replication, exploratory status and remaining scope.
