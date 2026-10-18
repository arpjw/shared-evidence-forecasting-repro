# Shared Evidence Forecasting

A reproducible research project about aggregating probabilistic forecasts when
forecasters share evidence. The companion Writing article is for
[aryasomu.com](https://aryasomu.com/writing/), built around an exact model and
controlled experiments. The project includes a tested simulator, a frozen
synthetic primary evaluation, an independently replayable question ledger,
and a replicate-level analysis. It contains no empirical claims about AI or
human forecasters.

## Question

If individually calibrated forecasters agree, how much independent information
does their agreement represent? Can a discount for shared evidence be learned
from a modest number of resolved questions?

## Initial model

For a binary outcome with a 50% prior, forecaster `i` observes a Gaussian
signal `S_i = a(2Y-1) + sqrt(rho)U + sqrt(1-rho)E_i`, where `U` is common
noise and each `E_i` is private noise. Its individually correct posterior log
odds is `2aS_i`. The log odds from all signals are the sum of individual log
odds divided by `1 + (n - 1)rho`. The effective number of independent signals
is `n / (1 + (n - 1)rho)`.

These formulas assume equal signal strength, equal marginal noise variance,
exchangeable overlap, and a 50% prior. Later phases test learned discounts and
independent source groups.

## Local setup

Python 3.11 or later is sufficient. The current core and tests use only the
standard library. A local `.venv` is intentionally excluded from Git.

```sh
cd shared-evidence-forecasting-repro
python3 -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m shared_evidence.experiment \
  --config configs/development-pilot-v0.json \
  --output results/development-pilot-v0.json \
  --ledger results/development-pilot-v0.jsonl.gz
.venv/bin/python -m shared_evidence.analysis \
  --input results/development-pilot-v0.json \
  --output results/development-pilot-v0-analysis.json
```

The `results/` directory is ignored by Git. The pilot config and run output
are development diagnostics. The output stores
the configuration, Python version, and hashes of the code that produced it.
`configs/development-groups-v0.json` exercises three source-group partitions
under a separate phase namespace and seed.
`configs/development-precision-v0.json` uses 30 fresh training and test
replicates for a runtime and uncertainty diagnostic. An optional compressed
ledger saves every generated question and test prediction with hashes; its
replay check regenerates questions from seeds and recomputes scores.
The primary evaluation was frozen in
[`configs/production-primary-v1.manifest.json`](configs/production-primary-v1.manifest.json)
before its run. Recreate it with:

```sh
.venv/bin/python -m shared_evidence.experiment \
  --config configs/production-primary-v1.json \
  --manifest configs/production-primary-v1.manifest.json \
  --output results/production-primary-v1.json \
  --ledger results/production-primary-v1.jsonl.gz
.venv/bin/python -m shared_evidence.verify \
  --summary results/production-primary-v1.json \
  --ledger results/production-primary-v1.jsonl.gz \
  --manifest configs/production-primary-v1.manifest.json
.venv/bin/python -m shared_evidence.analysis \
  --input results/production-primary-v1.json \
  --output results/production-primary-v1-analysis.json \
  --manifest configs/production-primary-v1.manifest.json \
  --ledger results/production-primary-v1.jsonl.gz
```

The original locked run completed all 30 training and test repetitions, with
200 training and 10,000 test questions per repetition. Independent replay
verified 306,000 question rows, 1,800,000 test predictions, all fitted
discounts, and all summary scores. The learned pool's mean held-out log loss
was 0.51029, versus 0.54316 for the arithmetic mean and 1.17889 for the
independence pool. Its paired mean differences were -0.03287 and -0.66860
respectively; the prespecified 97.5% bootstrap intervals were
[-0.03452, -0.03085] and [-0.67694, -0.66031]. Negative differences favor
the learned pool. These results concern only the declared Gaussian simulator.
The ignored `results/` files include the 94 MB compressed ledger. Its exact
bytes are also attached to the public
[primary-v1 release](https://github.com/arpjw/shared-evidence-forecasting-repro/releases/tag/primary-v1).
See
[`docs/research_protocol.md`](docs/research_protocol.md) for design and limits,
[`docs/validation_report.md`](docs/validation_report.md) for receipts and
checks, and [`reports/production-primary-v1-analysis.json`](reports/production-primary-v1-analysis.json)
for all saved statistics.

## Exploratory stress suite

After inspecting the primary result, we froze a separate exploratory suite
with seven configurations: 192 condition and training-size cells, 1,728 fitted
models, and 2,304,000 held-out questions. Every saved result passed full
question-ledger replay. The suite studies a limit of scalar discounting:
unequal independent source groups may require different weights, and one
nonnegative multiplier cannot reverse the sign of the summed forecasts.

The [stress protocol](docs/stress_protocol.md),
[suite lock](configs/stress-suite-v1.lock.json),
[validation report](docs/stress_validation_report.md),
[complete descriptive table](reports/stress-suite-v1.json), and
[Writing draft](docs/writing_draft.md) are included. The full stress ledgers
remain in ignored `results/stress-v1/` and are preserved in the public
[stress-v1 release](https://github.com/arpjw/shared-evidence-forecasting-repro/releases/tag/stress-v1).
The validation report records their SHA256 digests. Figures are generated from the archived reports with
`viz/make_figures.py`. Install the optional plotting dependency with
`.venv/bin/python -m pip install -r requirements-viz.txt`.

## Learning source groups

A separate frozen synthetic follow-up asks whether stable independent source
groups can be inferred from resolved forecasts. It removes the common outcome
signal by multiplying each training logit by the resolved outcome sign, then
clusters the remaining conditional correlations. It estimates overlap and fits
one final confidence multiplier using only the same training questions as the
scalar comparator. The oracle alone receives the planted group labels.

The [source-inference protocol](docs/source_inference_protocol.md),
[suite lock](configs/source-inference-suite-v1.lock.json),
[validation report](docs/source_inference_validation_report.md), and
[complete analysis](reports/source-inference-suite-v1.json) document the new
phase. In the fresh 9+1, overlap .9 primary cell, inferred-source pooling had
mean test log loss 0.48351, versus 0.55071 for the scalar learner and 0.48288
for the oracle. Both prespecified paired-interval gates passed. This result is
limited to the declared Gaussian simulator; the sensitivity grid retains
small-sample failures and unidentifiable zero-overlap cells.

To replay the final suite, use Python 3.11 or later in the same runtime version
recorded by the lock. The four compressed ledgers are attached to the
[source-inference-v1 release](https://github.com/arpjw/shared-evidence-forecasting-repro/releases/tag/source-inference-v1).
They belong under ignored `results/` and need to be restored if absent:

```sh
for name in primary sensitivity-m100 sensitivity-m400 sensitivity-m2000; do
  .venv/bin/python -m shared_evidence.source_inference_experiment \
    --config "configs/source-inference-${name}-v1.json" \
    --ledger "results/source-inference-${name}-v1.jsonl.gz" \
    --output "results/source-inference-${name}-v1.json" \
    --lock configs/source-inference-suite-v1.lock.json
  .venv/bin/python -m shared_evidence.source_inference_verify \
    --summary "results/source-inference-${name}-v1.json" \
    --ledger "results/source-inference-${name}-v1.jsonl.gz" \
    --lock configs/source-inference-suite-v1.lock.json \
    --output "results/source-inference-${name}-v1-verification.json"
done
.venv/bin/python -m shared_evidence.source_inference_analysis \
  --input results/source-inference-primary-v1.json \
  --verification results/source-inference-primary-v1-verification.json \
  --input results/source-inference-sensitivity-m100-v1.json \
  --verification results/source-inference-sensitivity-m100-v1-verification.json \
  --input results/source-inference-sensitivity-m400-v1.json \
  --verification results/source-inference-sensitivity-m400-v1-verification.json \
  --input results/source-inference-sensitivity-m2000-v1.json \
  --verification results/source-inference-sensitivity-m2000-v1-verification.json \
  --lock configs/source-inference-suite-v1.lock.json \
  --output reports/source-inference-suite-v1.json
```

## Current structure

- `shared_evidence/model.py`: stable logistic conversion and exact aggregation
  for the exchangeable model.
- `shared_evidence/groups.py`: exact pooling for independent source groups.
- `shared_evidence/simulate.py`: deterministic binary questions and signals.
- `shared_evidence/evaluate.py`: baselines, proper scores, and training-only
  discount fitting.
- `shared_evidence/experiment.py`: development and locked production runner.
- `shared_evidence/analysis.py`: paired, replicate-level summaries and
  deterministic bootstrap intervals.
- `shared_evidence/ledger.py`, `shared_evidence/archive.py`, and
  `shared_evidence/verify.py`: compressed question records, integrity receipts,
  and simulator replay.
- `shared_evidence/freeze.py`: pre-run manifest and source binding.
- `scripts/stress_lock.py`, `scripts/stress_report.py`, and
  `scripts/group_signs.py`: exploratory provenance, full-grid reporting, and
  grouped decision diagnostics.
- `viz/make_figures.py`: reproducible article figures from the saved reports.
- `shared_evidence/source_inference.py`: training-only source-group estimation.
- `shared_evidence/source_inference_experiment.py`,
  `shared_evidence/source_inference_verify.py`, and
  `shared_evidence/source_inference_analysis.py`: the frozen follow-up runner,
  independent full-ledger replay, and paired analysis.
- `scripts/source_inference_lock.py`: pre-run binding of the complete new suite.
- `viz/source_inference_figures.py`: plot from the verified follow-up report.
- `tests/`: mathematical, distributional, numerical, and integration checks.

The stress suite is exploratory, with only eight or twelve independent fits
per cell and no multiplicity adjustment across cells. The later source-inference
primary test is a fresh synthetic follow-up, not a replication of the original
primary question. None of these experiments establishes performance on human
or AI forecasts. The companion Writing article presents these results in
conversational form.

## Public snapshot

This repository is a clean export of the validated `source-inference-v1` research state. It has no private repository history. Full compressed synthetic ledgers are attached to the `primary-v1`, `stress-v1`, and `source-inference-v1` releases. Their byte sizes and SHA256 digests are in [`RELEASE_ASSETS.json`](RELEASE_ASSETS.json). The same verifier code and frozen configurations are here.

The repository is published without a reuse license unless a license is added explicitly.
