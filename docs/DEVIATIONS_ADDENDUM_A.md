# Deviations addendum A — feature-tokenised transformer and alignment-based conformance

**Scope.** This addendum covers only the two blocks of work run on 2026-09-20 under
`code_run/e22_ftt.py` and `code_run/e23_alignments.py`, whose outputs are `results/ftt_*` and
`results/conf_align_*`. It adds to `DEVIATIONS.md`; it replaces nothing in it. Where it *retires*
a deviation already logged there, it says so and names it, but it does not edit that file.

Everything below is a real computed quantity or a statement that a quantity was not computed.
No near-neighbour quantity is reported under another quantity's name.

---

## A.1 Substitution 2.1 (sklearn MLP in place of the FTT) is retired where marked, and only there

`DEVIATIONS.md` substitution 2.1 and the docstring of `common.FTTLite` both state that PyTorch is
not installed in the run environment and that the FTT row is therefore a scikit-learn
`MLPClassifier`.

**That statement is no longer true of this container.** `torch` 2.14.0 imports and runs:
`torch.cuda.is_available()` is `False`, so it runs on CPU, and a forward/backward pass completes
normally. `pip show torch` reports it installed at `/usr/local/lib/python3.11/dist-packages`, with
a directory mtime of **2026-09-20 10:15:06**; a `pip install torch --index-url
https://download.pytorch.org/whl/cpu` issued at the start of this run reported every requirement
already satisfied and installed nothing. The wheel that is present is the default PyPI build
(`2.14.0+cu130`), not the `+cpu` build, but it runs on CPU because no GPU is visible. In other
words torch became available at some point during the 2026-09-20 session, after the scripts that
record its absence were written, and the absence claim was simply never revisited.

`code_run/e22_ftt.py` therefore implements the real architecture. `common.FTTLite` is untouched and
every result file it produced is untouched: the MLP numbers stand where they are, under the name
they have, and the FTT numbers are written to separate files.

### A.1.1 What was implemented

Gorishniy, Rubachev, Khrulkov & Babenko (NeurIPS 2021), §3.1–3.2:

* **Feature tokeniser.** Each of the *p* features becomes its own token in R^d.
  Numerical feature *j*: `T_j = x_j · W_j + b_j` with `W_j, b_j ∈ R^d`, one learned vector per
  feature. Categorical feature *j*: `T_j = E_j[c_j] + b_j` with `E_j` a per-feature embedding
  table. (The tables are laid out contiguously inside one `nn.Embedding` with per-feature offsets
  — arithmetically identical to one table per feature, one kernel call instead of *n_cat*.)
* **A learned [CLS] token** prepended, giving *p*+1 tokens.
* **L pre-norm transformer blocks**: LayerNorm → multi-head self-attention → residual;
  LayerNorm → feed-forward (d → 2d → d, ReLU) → residual.
* **Head on [CLS] only**: LayerNorm → ReLU → Linear(d, 1).

This is an FTT and not an MLP: per-feature tokenisation, a [CLS] token, and self-attention between
feature tokens. It is small and it is not tuned; no claim is made that it is state of the art.

### A.1.2 Protocol held fixed

Identical to `e1_dissociation.py`, `e14_baf_dissociation.py` and `e20_ieee_dissociation.py`: the
same splits (read from those scripts, not re-invented), the same five seeds `[0,1,2,3,4]`,
value-weighted recall at a 1% false-positive rate with count-based recall and PR-AUC beside it.
Class imbalance is handled by `pos_weight = n_neg/n_pos` in the BCE loss, the loss-side equivalent
of the `scale_pos_weight` LightGBM is given in `common.fit_gbdt`.

### A.1.3 New deviations this block introduces

**A.1.3-a — restricted training budget (all arms).** Two cores, no GPU, shared with a second agent.
Measured throughput varied by up to an order of magnitude within the run depending on contention.
The epoch budgets actually used are recorded per arm in `results/ftt_config.json`; the per-epoch
training loss is in `logs/e22_ftt.log` (seed 0: Vehicle 1.294 -> 0.875 over 8 epochs, still
falling; BAF 0.964 -> 0.810 over 2 epochs, still falling; Medicare 0.5114 -> 0.5089 between its
last two of 20, flat). **On the Vehicle and BAF arms the training loss was still falling when the
budget ran out**, so those two FTT numbers are a lower bound on what this architecture reaches on
those data and must not be read as "the transformer is worse than the tree ensemble" without that
qualification. The Medicare arm's loss had flattened.

**A.1.3-b — IEEE-CIS FTT not computed.** Self-attention is quadratic in the number of feature
tokens and IEEE-CIS has 432 features. Measured on this machine: **19 training rows/s** on one
contended thread and **320 training rows/s** on two uncontended threads, at d=16, 2 heads, 2 blocks,
batch 256. Scoring alone requires a forward pass over the 147,635 test rows for each of five seeds.
Even on the optimistic figure the arm needs roughly an hour of exclusive CPU; on the pessimistic one
it needs a day. It was not run. **The IEEE-CIS FTT cell stays open.** The MLP number already in
`tab_dissociation_fraud_summary_IEEE.csv` (0.2309) is an MLP number and must not be relabelled.

**A.1.3-c — ULB FTT not computed.** Out of time, not out of reach; the arm is implemented
(`e22_ftt.py ulb`) and costs roughly twenty minutes of exclusive CPU.

**A.1.3-d — leakage-side FTT not computed.** The FTT row of Table 3 also has three leakage cells.
They were not run. Only the fraud side of the FTT row is addressed here.

**A.1.3-e — numerical preprocessing.** Gorishniy et al. use a quantile transform on numerical
features. This run standardises (train mean and SD) and clips to ±10. Recorded, not tuned.

---

## A.2 Alignment-based conformance: a declared variant sample, not an estimator substitution

`DEVIATIONS.md` 7.6 logs token-based replay as a **substitution** for optimal alignment, and that is
why the CONF, GBDT+φ and H1(b) cells stayed open: token replay is a different estimator from the
one §4.2 and Supplementary S2 specify.

`code_run/e23_alignments.py` **keeps the estimator** — PM4Py A* optimal alignment, unit cost for a
log move and for a model move, zero for a synchronous move — against **the same** normative model:
the same reference set (cases of the flow type whose trace ends in `Clear Invoice` with no
correction activity), the same Inductive Miner (infrequent), the same primary noise threshold 0.2,
discovered exactly as `e11_conformance.py` and `e17_conformance_replay.py` discover it. What is
restricted is the **population**, not the estimator.

### A.2.1 Sample design (this is the deviation)

Recorded per flow type in `results/conf_align_frame.csv`.

* **Sampling frame**: every distinct trace variant of the flow type, with its case frequency.
* **Stratum C (certainty / take-all)**: the `N_CERTAINTY` most frequent variants, inclusion
  probability 1.
* **Stratum R (random)**: a simple random sample without replacement from the remaining variants,
  drawn with `numpy.random.default_rng(0)` — **seed 0**. Designed inclusion probability
  `N_RANDOM / n_tail`, recorded per flow.
* Stratum R is attempted **in the seeded random order** under a wall-clock budget. Variants the
  budget does not reach are recorded as not attempted. Because the order is a uniformly random
  permutation, the attempted set is itself a simple random sample of the tail, and its **realised**
  inclusion probability is recorded beside the designed one.
* **Per-variant timeout** (PM4Py `max_align_time_trace`). A variant that exceeds it is **censored**:
  counted, written with its length and case frequency to `results/conf_align_censored.csv`, and
  contributing no alignment cost. No variant is dropped without record.

### A.2.2 What the sample costs, and the bias it carries

Coverage is reported three ways in `conf_align_frame.csv`: share of variants aligned, share of the
flow's **cases** represented, and share of the flow's **outcome-labelled positives** represented.

The third matters and is the honest weak point of this block. Leakage positives carry correction
activities, so they sit in long, rare trace variants, which are exactly the variants a
frequency-weighted certainty stratum misses and exactly the variants that time out. **The evaluated
subpopulation therefore has a materially lower positive rate than its flow type**, and both rates
are recorded per flow. The detection numbers below are properties of the evaluated subpopulation —
the cases whose variant was successfully aligned — and not of the flow type. They should be
typeset with that restriction stated.

The H1(b) contrast is nevertheless a fair paired comparison, because CONF and GBDT+φ are scored on
exactly the same cases, the same five seeds and the same 70/30 stratified split.

### A.2.3 Detection protocol

Identical to `e17_conformance_replay.py`, so the alignment estimator and the replay estimator are
directly comparable:

| detector | input |
|---|---|
| CONF (alignments) | the alignment cost δ |
| GBDT+φ (alignments) | record features **plus** φ(σ) = (alignment cost, number of log moves, number of model moves, trace length, six typed-deviation indicators) |
| GBDT (records), IF, AE | record features only, exactly as in `e17` |

Five seeds, stratified 70/30 split per seed, value-weighted recall at a 1% false-positive rate with
count-based recall beside it. H1(b) is the paired difference CONF − GBDT+φ.

### A.2.4 What is still not computed on the leakage side

* **Rules and TSEQ on the evaluated subpopulation.** `e23_alignments.py` runs CONF, GBDT+φ,
  GBDT (records), IF and AE. It does not run the rules detector or the next-event sequence model.
  The H1(a) sentence ("the best of the rules, the isolation forest, the autoencoder and the
  trace-sequence model") therefore **cannot** be completed from this run, and the two slots it
  needs stay open. The IF and AE numbers on the evaluated subpopulation are reported in
  `conf_align_detection_summary.csv` as partial input to it and nothing more.
* **The noise-threshold grid.** Only the primary threshold 0.2 was aligned. Fitness, precision and
  the recall lost at threshold 0.4 are not computed from alignments here.
* **The 2-way match flow.** All 148 of its variants align, but the flow carries only 4
  outcome-labelled positives in total, so its detection cells are not evaluable. Reported as such,
  not left blank.
* **Cost-weight sensitivity** (Spearman correlation of δ rankings under move weights {0.5, 1, 2}),
  which S2 specifies, was not run.

### A.2.5 Comparability with the existing leakage cells

The leakage cells already in `main_results.tex` (Rules, IF, AE, TSEQ, GBDT records for
3-way-after-GR) come from `tab_dissociation_leakage_raw.csv`, whose evaluation set is the 30% test
split of the **whole** flow (n = 4,555, 205 positives). The alignment cells come from the 30% test
split of the **evaluated subpopulation**. These are different populations. Putting a CONF cell from
this run beside an IF cell from that run in one table column is not a like-for-like comparison, and
the table note should say so, or the whole column should be rebuilt from
`conf_align_detection_raw.csv`, which contains IF, AE and GBDT (records) on the alignment
population for exactly that purpose.

### A.2.6 Ablation: is GBDT+φ winning on alignment features or on the typed deviations?

φ(σ) as `e17_conformance_replay.py` builds it, and as `e23_alignments.py` copies it for
comparability, contains the four alignment-derived quantities **and** the six typed-deviation
indicators `T_*`. Those indicators are the conformance channel's *deviation typology*, and §4.2
defines φ(σ) as activity counts, directly-follows relations and alignment moves — not as the
typology. If GBDT+φ were winning on the `T_*` columns alone, the H1(b) result would be an artefact
of the feature set rather than evidence about the two channels.

It is not. `results/conf_align_phi_ablation_{raw,summary,tests}.csv`, five seeds, same metric:

| flow type | GBDT+φ, alignment moves only | GBDT+φ, typed deviations only | GBDT+φ, full |
|---|---|---|---|
| 3-way, invoice after GR | 99.6% | 97.3% | 97.3% |
| 3-way, invoice before GR | 94.7% | 90.4% | 95.1% |

The alignment-derived features alone carry the result: CONF − GBDT+φ(alignment moves only) is
**−50.5 pp** (p = 0.012) on invoice-after-GR and **−87.8 pp** (p = 1.4 × 10⁻⁶) on
invoice-before-GR. H1(b) is negative whichever version of φ is used.

**One caveat on this table.** The ablation script re-derives its 70/30 splits from the row order of
`conf_align_delta.csv` rather than from the sorted case index `e23_alignments.py` uses, so its
seeds land on different splits of the same population. That is why its CONF column reads 49.1% and
6.9% where `conf_align_detection_summary.csv` reads 46.0% and 6.1%. **The `e23` numbers are the
canonical ones**; the ablation table is for the relative comparison between the three versions of
φ and for nothing else.
