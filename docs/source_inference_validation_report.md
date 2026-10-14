# Source inference v1 validation

This is a frozen synthetic follow-up designed after the original primary result
and exploratory stress suite. It tests learning stable source groups from
resolved forecasts in the declared Gaussian simulator. It is not evidence about
human or AI forecasting teams.

## Frozen design and checks

The [protocol](source_inference_protocol.md), four final configurations, Python
version, simulator, learner, runner, scoring, independent verifier, analysis,
ledger code, and lock code were bound in the
[suite lock](../configs/source-inference-suite-v1.lock.json) before final data
generation. Lock SHA256:

`67c8cd2c92cbddc4f82aedb39cabdb4ad0ad9493ed9348d95ec64ba1d6dfafe1`

The full test suite passed 79 tests before the freeze. A development pilot
generated 400 questions and 1,200 predictions, and passed independent replay;
its outcomes were not used to change the fixed learner. All four final runs
completed without source or configuration drift. Independent replay regenerated
every saved question, recomputed all 462 fits from training forecasts and
outcomes, and recomputed 6,084,000 test predictions and scores. The four
ledgers cover 1,386,000 generated questions, including 1,014,000 test
questions.

| Phase | Fits | Questions replayed | Test predictions | Ledger SHA256 |
| --- | ---: | ---: | ---: | --- |
| Primary | 30 | 162,000 | 900,000 | `e8370499a5fbbdd2ded61939a1eccb94a4f608172a124deed1ce8857f6767918` |
| Sensitivity M=100 | 144 | 302,400 | 1,728,000 | `90c0c52a8f801e9723e5a7746db6b96b777259029d87c057e5544e31b2ba5ac6` |
| Sensitivity M=400 | 144 | 345,600 | 1,728,000 | `0f32c745d0fb3dffc16f832dd3fc5b0a2abdc2bdc93fc923ba712a4cf4a1e282` |
| Sensitivity M=2000 | 144 | 576,000 | 1,728,000 | `ccc3c3bf46a4206b92157f87e076ea730cfd25554f2be0eb06c6c87acccf4186` |

The checked-in [analysis report](../reports/source-inference-suite-v1.json)
has SHA256
`b19c485cd28037f8de9914d6560e5d03891ef40b0382dd4e41320a46d5a681c3`.
The complete compressed ledgers and summaries are ignored local files under
`results/`; they must be retained separately for full byte-level replay.
The report records each verifier receipt, including summary and ledger hashes.

## Prespecified primary result

For the 9+1 partition, overlap .9, 400 training questions per fit, 30 fits,
and 5,000 new test questions per fit, mean held-out log loss was 0.550710 for
the scalar learner, 0.483506 for inferred source groups, and 0.482884 for the
group oracle. The paired learned-minus-scalar difference was -0.067204 nats,
with a prespecified 97.5% replicate-bootstrap interval
[-0.069405, -0.065002]. Learned-minus-oracle regret was 0.000622 nats, with
the corresponding interval [0.000239, 0.001115]. Both declared gates passed:
the first interval is below zero, and the second has upper endpoint below
0.01 nats. These approximate intervals resample fitted replicates, not
individual questions.

The learner recovered the planted partition in all 30 observed primary fits.
Its mean estimated overlap was 0.8992. It closed 99.08% of the mean
scalar-oracle log-loss gap, a descriptive ratio without its own confidence
interval. The scalar and oracle logits had opposite strict signs on 25,222 of
150,000 test questions. The inferred learner matched the oracle's side on
25,153 of those questions (99.73%). This is oracle-direction agreement, not
observed classification accuracy.

## Sensitivity and limits

Across the 36 descriptive sensitivity cells, source recovery is not defined
at overlap zero: every partition produces the same observable distribution.
There, inferred and scalar scores were exactly equal in all nine cells. Among
the 324 positive-overlap sensitivity fits, the declared partition was recovered
in 323. The one failure used 100 training questions, partition 4+3+2+1 and
overlap .5; it split the three-member source into singletons. That fit's
oracle regret was 0.035710 nats, and its cell's mean regret was 0.006886
nats with 11/12 exact recovery. Several other small-training fits recovered
the groups but lost performance through calibration-slope estimation.

Equal 5+5 groups have equal optimal weights; a scalar can represent their
source-aware pool. Small observed gains from the inferred learner in those
cells reflect finite-sample fitting constraints, not a structural need for
asymmetric weights. One cell had slightly negative realized oracle regret from
test-sample variation; it does not beat the population Bayes rule. Sensitivity
intervals are descriptive 95% intervals without a grid-wide multiplicity
claim. Training-size streams are independently seeded and need not improve
monotonically.

The method assumes stable forecaster identities, fixed disjoint groups, equal
individual signal strength, symmetric Gaussian noise, a shared 50% prior,
complete forecasts, and no distribution shift. Conditional dependence in real
forecast data need not identify a shared causal source. The fixed graph
threshold can merge groups through a noisy bridge or split a weak source.

## Reproduction

Run the four `source-inference-*.json` final configurations through
`shared_evidence.source_inference_experiment` with the suite lock, then run
`shared_evidence.source_inference_verify` on each summary and ledger. Supply
all four verified summaries and receipts to
`shared_evidence.source_inference_analysis`. Exact commands are in the
[README](../README.md). The figure script
[`viz/source_inference_figures.py`](../viz/source_inference_figures.py)
generates the new article figure from the checked-in verified report; its
input and output hashes are in
[`figures/source_inference_figure_manifest.json`](../figures/source_inference_figure_manifest.json).
