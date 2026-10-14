# Primary synthetic evaluation validation

Date: 2026-10-14. Scope: the frozen primary cell only, with 10 forecasters,
conditional overlap 0.5, signal strength 0.5, 200 training questions, 10,000
held-out questions, and 30 independent repetitions.

## Freeze and artifacts

- Frozen manifest: `configs/production-primary-v1.manifest.json`, SHA256
  `afd6f762fd2bcde9798fde9358b56027257e91ce46fabc905661a53523fce552`.
- Summary: `reports/production-primary-v1-summary.json`, SHA256
  `f2dec062747dce7ff114b1ed103536ec54ef4293c4bb7363e48c3f419345e4a2`.
- Analysis: `reports/production-primary-v1-analysis.json`, SHA256
  `79fbae86837d4ec5cfa097440ff5de792f3a13877411b43edcb431386d68800b`.
- Local ledger: `results/production-primary-v1.jsonl.gz`, SHA256
  `b10e6512a35ab960e6cb96f02db61b174888508b04f48848f531cd06e548f323`.
  It is 94 MB compressed and excluded from Git. The summary contains its byte
  receipt. The reported replay requires this exact ledger or a complete
  deterministic rerun under the frozen Python and source versions.

The analysis input digest matches the summary bytes, the analysis manifest
digest matches the frozen manifest bytes, and the summary ledger receipt
matches the local compressed ledger bytes.

## Validation performed

- `python -m unittest discover -s tests -q`: 49 tests passed before the run.
  Coverage includes independent Gaussian oracle checks, moments, symmetry,
  optimizer behavior, seed separation, serialization, replay, and rejection of
  forged stream descriptors, missing source hashes, and unrelated frozen
  configs.
- Locked verifier: 306,000 question rows, 1,800,000 test predictions, and
  all 30 fitted discounts and aggregate scores reconciled.
- Locked analysis: 10,000 seeded percentile bootstrap resamples of paired
  repetition means, with shared repetition draws for the two primary contrasts.

## Primary result

Mean held-out log loss in nats per question was 0.510290 for the learned pool,
0.543158 for the arithmetic mean, 1.178886 for the independence pool, and
0.507389 for the parameter-aware oracle. The learned-minus-arithmetic-mean
paired difference was -0.032868 with prespecified 97.5% interval
[-0.034517, -0.030849]. The learned-minus-independence difference was
-0.668596 with 97.5% interval [-0.676944, -0.660309]. Negative differences
favor the learned pool. These intervals apply to the two prespecified primary
contrasts only; the method-score and Brier intervals in the analysis file are
descriptive 95% intervals.

The result is evidence for a one-parameter discount in this declared Gaussian
simulator. It does not establish performance on real forecasters, language
models, or information sources. The broader sensitivity grid and production
source-group stress test have not been run. The percentile bootstrap has
approximate coverage with 30 repetitions, and no claim of exact finite-sample
familywise error control follows from it.
