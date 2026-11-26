# The retrained arm of the durability experiment: the fix, the run, and what it supports

Written 2026-09-21 (WP-G). Scope: `code/e4_durability.py`, the Medicare arm of H3(b), and the two
Results slots that depend on it. The frozen arm and the primary endpoint are untouched and were
re-verified against the files on disk.

## 1. The defect and the repair

`DEVIATIONS.md` item 8.7 records the defect. The loop of 2026-09-19 let the adversary move, appended
the resulting round-(t+1) records to the training set, refitted the detector, and then scored those
same records at the top of the next iteration. The retrained detector was evaluated on the exact
records it had just been fitted on.

The Methods sentence the repair follows, quoted from the "Adaptive adversary" subsection of
`latex/main_submission.tex`:

> In round $t$ the adversary observes the deployed detector $f_{t-1}$ through score queries only,
> with no gradient access, and solves (2) for each fraudulent case by coordinate descent within a
> density-based plausibility set [...] The institution either retrains $f_t$ on the labelled
> outcomes of round $t$ after a labelling delay $\Delta$, or does not retrain at all, which is the
> frozen detector of Hypothesis 3(a); a round is one labelling delay, because the adversary can
> respond only to what has been caught and labelled.

Read literally, the detector deployed in round $t$ is $f_{t-1}$, and the records of round $t$ are
labelled one round later, so the detector fitted on them is deployed from round $t+1$. The repaired
loop therefore refits before the adversary moves. The order at round $t$ is: the deployed detector
$f_{t-1}$ scores the round-$t$ records; those records are labelled and fit $f_t$; the adversary of
round $t+1$ best-responds to $f_t$; $f_t$ scores the new round-$(t+1)$ records. Round 0 carries no
adversarial records, so the first refit uses the round-1 records and $f_1$ is first deployed at
round 2. Rounds 0 and 1 are consequently identical in the two arms by construction.

The repair is switchable. `--retrain-eval legacy` reproduces the 2026-09-19 order exactly and keeps
the original file names; `--retrain-eval fixed`, the default, applies the order above and writes
every output under the prefix `fig5_durability_retrained_fixed_`, so no file of the earlier run is
overwritten. Seeds, the q grid, the kappa grid, the delta grid and the round count are unchanged.
The frozen arm runs neither branch, so its call sequence is the one of 2026-09-19.

## 2. The command

```
cd code
python3 e4_durability.py --retrain-eval fixed --datasets medicare --cells cross
```

36 seconds of wall clock. Outputs, all new files in `results/`:

| file | contents |
|---|---|
| `fig5_durability_retrained_fixed_raw.csv` | every round of every cell, both arms, with the instrumentation of section 4 |
| `fig5_durability_retrained_fixed_h3_primary_endpoint.csv` | $V_T/V_0$, $R_0$, $R_T$ and monotonicity at the primary cell |
| `fig5_durability_retrained_fixed_h3_spread_by_free_quantity.csv` | last-round spread across q, kappa and the retraining switch |
| `fig5_durability_retrained_fixed_h3_leakage_side.json` | the unchanged leakage-side statement |
| `fig5_durability_retrained_fixed_config.json` | the protocol, the Methods sentence followed, and the command |
| `fig5_durability_retrained_fixed_partial_Medicare.csv` | the per-cell partial dump |

The grid is the S4 cross at the primary cell with three seeds (0, 1, 2) and one seed elsewhere,
which is the configuration `results/fig5_durability_raw.csv` and `results/h3_primary_endpoint.csv`
record. The run also reached q = 0.25 and the two kappa cells, which the 2026-09-19 run did not.

## 3. Verification of the frozen arm

All 55 frozen rows that the two files share are equal to the last stored digit in value-weighted
recall, count recall and value stolen. The primary endpoint file agrees:

| quantity | on disk (`h3_primary_endpoint.csv`) | this run | printed |
|---|---|---|---|
| $R_0$, frozen | 0.7039300897393844 | 0.7039300897393844 | 70.4% |
| $R_T$, frozen | 0.1908238166718441 | 0.1908238166718441 | 19.1% |
| $V_T/V_0$, frozen | 2.7640330708567693 | 2.7640330708567693 | 2.76 |

Nothing quoted in Results or Discussion from the frozen arm changes.

The switch itself was checked as well. Running `--retrain-eval legacy --datasets medicare --cells
primary` in a copy of the tree outside the paper folder reproduces all 66 rows of the primary cell,
both arms, to the last stored digit of `results/fig5_durability_raw.csv`. The legacy path is
therefore preserved exactly and the difference in section 4 comes from the evaluation order alone.
No result file of the earlier run was written in the paper folder.

## 4. The two paths, Medicare, q = 0.05, kappa = 0.1, three seeds

Value-weighted fraud recall at a 1% false-positive rate, mean +- SD over seeds 0, 1 and 2. The
frozen column is reproduced from this run and equals the file on disk. The legacy column is the
in-sample retrained path of 2026-09-19, shown for comparison only. Source for the first three
columns: `results/fig5_durability_retrained_fixed_raw.csv`; for the legacy column:
`results/fig5_durability_raw.csv`.

| round | frozen | retrained (fixed) | retrained (legacy) | fixed minus frozen |
|---:|---|---|---|---:|
| 0 | 70.4 +- 4.0 | 70.4 +- 4.0 | 70.4 | +0.0 |
| 1 | 19.7 +- 0.6 | 19.7 +- 0.6 | 31.2 | +0.0 |
| 2 | 19.4 +- 0.6 | 31.2 +- 11.1 | 19.4 | +11.8 |
| 3 | 19.1 +- 0.0 | 19.4 +- 8.5 | 24.7 | +0.3 |
| 4 | 19.1 +- 0.0 | 24.7 +- 12.1 | 36.2 | +5.6 |
| 5 | 19.1 +- 0.0 | 36.2 +- 17.0 | 23.0 | +17.1 |
| 6 | 19.1 +- 0.0 | 23.0 +- 14.2 | 49.7 | +3.9 |
| 7 | 19.1 +- 0.0 | 49.7 +- 18.8 | 50.9 | +30.6 |
| 8 | 19.1 +- 0.0 | 50.9 +- 12.8 | 48.5 | +31.8 |
| 9 | 19.1 +- 0.0 | 48.5 +- 27.0 | 41.6 | +29.4 |
| 10 | 19.1 +- 0.0 | 41.6 +- 17.3 | 42.3 | +22.5 |

Value stolen relative to round 0, mean over seeds, same source file: the frozen arm runs
1.00, 2.74, 2.75, 2.76 and then holds at 2.76 to round 10; the fixed retrained arm runs
1.00, 2.74, 2.32, 2.73, 2.54, 2.14, 2.60, 1.69, 1.64, 1.68, 1.95. At the last round the fixed
retrained $V_T/V_0$ is 1.95 +- 0.38 against the frozen 2.76 +- 0.35.

The fixed retrained path is the legacy path displaced by one round. That is not a coincidence and
it is the finding of the next section.

## 5. The repair is correct and it is not sufficient

The raw file now carries four instrumentation columns, written by the same loop that produces the
metrics: `n_moved_since_previous_round` counts the fraud records whose feature vector changed,
`n_scored_in_training_set` and `share_scored_in_training_set` count how many of the records the
deployed detector scores are records it was fitted on, and `mean_score_on_fraud_records` is the
detector's mean score on them. At the primary cell, means over the three seeds, retrained arm,
from `results/fig5_durability_retrained_fixed_raw.csv`:

| round | records moved (of 64) | share scored that are in the training set | mean detector score on them |
|---:|---:|---:|---:|
| 1 | 55.3 | 0.00 | 0.128 |
| 2 | 13.7 | 0.79 | 0.975 |
| 3 | 1.3 | 0.98 | 0.997 |
| 4 | 2.0 | 0.97 | 0.999 |
| 5 | 0.7 | 0.99 | 0.999 |
| 6 to 10 | 0.0 | 1.00 | 0.999 |

The adversary stops moving. Once the detector is fitted on a record, it scores that record at
almost 1.0, no move inside the plausibility set lowers the score, and the cost term
$\kappa \lVert x - x_0 \rVert_M$ then makes standing still the best response. From round 6 onward
the round-$(t+1)$ records are byte-identical to the round-$t$ records, so the detector is again
scoring its own training rows. Reordering the loop moves the evaluation from $f_t$ to $f_{t-1}$,
and $f_{t-1}$ was fitted on the same rows one round earlier, which is why the path is the legacy
path shifted by one.

The test window holds 64 fraud records, and the harness perturbs that one fixed set in every round
rather than drawing new fraud. An out-of-sample retrained arm needs a population the detector is
never fitted on, which means splitting the fraud records into a labelled half that feeds retraining
and a held-out half that is scored. That is a change of experiment rather than a change of
evaluation order, it is outside this package, and it has not been run.

## 6. The two Results slots

The sentence in Results is: "Retraining between rounds changes the shape rather than the level: the
path is \result{path shape under retraining} and it ends \result{retrained path against the frozen
path} the frozen one, which is what a repeated game with a labelling delay produces."

Taken at face value, the fixed run fills the slots as follows, every quantity from
`results/fig5_durability_retrained_fixed_raw.csv` and
`results/fig5_durability_retrained_fixed_h3_primary_endpoint.csv`:

- `path shape under retraining` = "not monotone, falling to 19.4% at round three and rising to 41.6% by round ten"
- `retrained path against the frozen path` = "22.5 percentage points above"

Two things stand against filling them.

First, the sentence's own claim fails on these numbers. The level moves as well as the shape. Recall
ends 22.5 percentage points above the frozen arm, and the primary endpoint moves from 2.76 to 1.95.
A sentence that says retraining changes the shape rather than the level is contradicted by the run
it would cite.

Second, the numbers are still in-sample, for the reason in section 5, and the seed spread is wide
enough that even their sign is unestablished. The paired difference at the last round is 22.5
percentage points with an SD of 17.3 over three seeds, which gives a paired t of 2.26 and p = 0.15.

The recommendation is therefore to leave both slots open and to recast the sentence. Proposed
recast, for WP-F to place:

> Retraining between rounds was run on the same grid, and its path is not reported. The adversary
> stops moving once the detector has been fitted on its records, so from the sixth round every
> record the retrained detector scores is a record in its own training set, and the resulting path
> measures memorisation rather than a repeated game (deviations log, item 8.7).

If the author prefers a sentence that carries a number, the defensible form states the limit rather
than the path:

> Retraining between rounds does not restore the level: recall against the retrained detector ends
> at 41.6% against 19.1% for the frozen detector, and that gap is an in-sample artefact, because the
> adversary stands still once the detector has been fitted on its records.

Both forms keep the frozen arm, which carries the primary endpoint, exactly as it is reported now.

## 7. Two consequences outside this package

The default of `--retrain-eval` is `fixed`, and `code/run_all.sh` calls `python3 e4_durability.py`
with no arguments. Under that default the run writes the prefixed files and no longer regenerates
`fig5_durability_raw.csv`, `h3_primary_endpoint.csv`, `h3_spread_by_free_quantity.csv` or
`h3_leakage_side.json`, which `fill_slots.py` reads. The line in `code/run_all.sh` needs
`--retrain-eval legacy` appended, or the four slots need repointing at the prefixed files. That file
belongs to another package and was not edited.

`code/e21_ieee_adversary.py` carries the same loop and the same defect, and it was not edited,
because it sits outside this package. The reordering of section 1 transfers to it line for line. No
IEEE-CIS retraining number appears in the manuscript, so nothing there is at stake, and the
saturation of section 5 would have to be checked separately on that file, where the fraud
population is far larger than 64 records.

`code/make_figures.py` was not changed. Figure 5 draws the frozen arm only, with a caption saying
the retrained arm is withheld because its evaluation was in-sample. Section 5 shows that this is
still true after the repair, so drawing the fixed retrained arm would put back in the figure what
the text withholds.
