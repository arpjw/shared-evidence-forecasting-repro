# Exploratory stress-suite validation

Date: 2026-10-14. The primary result was known before this suite was designed.
This is a descriptive sensitivity study, not a second confirmatory test. The
fixed design and code receipt is `configs/stress-suite-v1.lock.json`, SHA256
`25c5f13ec08251d091d14dc12f394885a3e3653046e01c37b7c761e1871455b0`.

## Coverage and replay

Seven prespecified configurations completed: three exchangeable training sizes
and four source-group training sizes. The saved report has 192 complete cells,
1,728 fits, 1,380,000 training question rows, and 2,304,000 held-out question
rows. Fits are independent across repetitions within a cell. Some conditions
share random draws, so 1,728 is not a count of mutually independent fits
across the suite. This clarifies the shorthand in the frozen protocol without
changing any design or analysis choice.

Every configuration passed `shared_evidence.verify` against its full compressed
ledger. In total that replay checked 3,684,000 question rows, 14,976,000
saved test predictions, all 1,728 training-only fitted discounts, and all
saved method scores. Four source-group ledgers also produced the prespecified
opposite-sign diagnostic. The locked report checked the seven config bytes,
source hashes, Python version, exact phase set, complete condition and
replicate coverage, and 10,000 descriptive bootstrap resamples. An independent
read-only audit recalculated all reported method and contrast means from the
raw summaries and exact bootstrap values for selected 9+1 cells.

The compact raw summaries and sign diagnostics are under `reports/stress-v1/`;
the complete derived table is `reports/stress-suite-v1.json`. The local full
ledgers are under ignored `results/stress-v1/`. Their byte identities are:

| Ledger | SHA256 |
| --- | --- |
| `stress-exchangeable-m1000-v1.jsonl.gz` | `b33ff21bb83fb54d0501f84a3865e4d0d45e9f1d658c56a4060aa0d01b8afcaa` |
| `stress-exchangeable-m200-v1.jsonl.gz` | `054d7c8537c54fac42fd50fe8d6767fb2bc19ae7a62c362e161bd2e72438ca5d` |
| `stress-exchangeable-m50-v1.jsonl.gz` | `cdf9c404be5988f876b2ade87b07a24ffc695790a4986be7937f8d6f578dfcf0` |
| `stress-groups-m1000-v1.jsonl.gz` | `c700f96d00e4d75794fb7bff0b35a79f1323fd882cd8fad175a5d7f70ce87b70` |
| `stress-groups-m200-v1.jsonl.gz` | `d2be04ab2839814e9a7f5876353fe734d553d2319413aec88bac322fb4c8c145` |
| `stress-groups-m50-v1.jsonl.gz` | `3b78b17e1b8c047e35dae5e3f1b22b0f9b881dd79d32b1756fc5471af2c61945` |
| `stress-groups-m5000-v1.jsonl.gz` | `47b08813cbfb358b354460b3557937779e50709749c4f704fea500c66f5ec267` |

The seven ledgers occupy about 902 MB locally. Their hashes also appear in
the respective raw summary receipts.

## Findings that survived the checks

The equal 5+5 source groups have no opposite-sign decisions for any tested
overlap; neither do the zero-overlap controls. In the 9+1 split at within-group
overlap 0.9 and M=5,000, 4,095 of 24,000 held-out questions (17.06%) gave
opposite decision signs for the summed reports and group-aware oracle. At
overlap 1, the count was 4,307 of 24,000 (17.95%). The latter was also
independently recomputed from raw ledger rows.

For the 9+1, overlap-0.9, M=5,000 cell, mean log loss was 0.551872 for the
learned scalar and 0.482295 for the group-aware oracle. The paired gap was
0.069577 nats per test question, with a descriptive 95% replicate-bootstrap
interval [0.066335, 0.072909]. More training reduced the finite-sample fit
noise but did not erase the structural gap. At overlap 1, the learned-minus-
oracle gap changed from 0.088790 at M=50 to 0.071572 at M=5,000, while its
population-optimal scalar approached 0.12195. At M=50 it scored worse than
the arithmetic mean in that cell; at M=5,000 it scored slightly better. These
different M values use distinct test streams, so this comparison describes
separate cells rather than a paired learning curve.

All stress intervals are descriptive and unadjusted across the 192 cells.
Exchangeable cells have eight fitted repetitions, group cells twelve. The
suite measures this Gaussian construction, not real human or AI forecasts.
Source labels and overlap are privileged inputs to the oracle; estimating
them is outside the experiment.

## Repair-loop record

No simulation, ledger, fit, or score validation failed in the seven locked
configurations. The one-off orchestration wrapper used a `stress-*.json` glob
and, after all seven runs passed, attempted to treat the newly saved lock file
as an eighth config. It failed before creating an eighth output. The wrapper
was corrected to enumerate exactly the seven config paths recorded in the
lock for reporting. No stress cell, source file, config, or locked outcome was
changed or rerun. This incident does not alter the seven valid run receipts.
