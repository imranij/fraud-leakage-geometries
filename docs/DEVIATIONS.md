# Deviations from the pre-specified design

Analyses executed 2026-09-19 and 2026-09-20 at the authors' decision, before OSF registration of the
pre-analysis plan; the manuscript's registration language must be revised accordingly.

Every departure below is from the design as stated in `main_journal.tex` and
`main_journal_supplement.tex` (S1–S4, S9) and in `code/configs/default.yaml`.

## 1. Datasets that could not be obtained

| # | Pre-specified | What happened | Consequence |
|---|---|---|---|
| 1.1 | BPI Challenge 2019 event log | **Resolved 2026-09-20**: obtained from 4TU through a browser on the author's machine and verified exactly (251,734 cases / 1,595,923 events). | The leakage-population description and Table S5, the E7 gate and H3(c) stationarity are now computed. H1(a) is computed for the record-only detectors; H1(b), the CONF row and the process-drift precision curve are **not** computed, for the compute reason in 4.1 below, not for want of data. |
| 1.2 | BPI Challenge 2017 | same host, same block | The process-generality check was not run. |
| 1.3 | IEEE-CIS | ~~Kaggle blocked; no mirror~~ **SUPERSEDED 2026-09-20: obtained, see section 8** | The IEEE-CIS column of the dissociation table and the restricted adversary on it are now computed; the H5 fraud cases still need participants. |
| 1.4 | Bank Account Fraud | **Resolved 2026-09-20**: the Base variant was fetched through a browser on the author's machine and verified exactly (1,000,000 rows x 32 columns, 11,029 positives = 1.1029%). | The BAF column of the dissociation table is now computed (section 7 below). The BAF **drift and adversary** runs were still not reached in the budget, and variants I--V were not extracted, so the six-variant bias analysis is not run. |
| 1.5 | PaySim | **Resolved 2026-09-20**: obtained and verified exactly (6,362,620 rows, canonical 11-column schema, 8,213 frauds = 0.129%). | The E7 gate and H4 were re-run on the host `configs/default.yaml` names; both hosts are reported (section 7 below). |

Seven of the eight named datasets have been obtained: IEEE-CIS (590,540 / 20,663), ULB (284,807 / 492), vehicle claims
(15,420 / 923), Medicare (5,410 providers / 506 flagged), BPI Challenge 2019 (251,734 cases /
1,595,923 events) and, later on 2026-09-20, Bank Account Fraud Base (1,000,000 / 11,029) and PaySim
(6,362,620 / 8,213), each verified exactly against Table 2. IEEE-CIS was obtained later the same day (section 8); BPI 2017 remains
unobtained.

## 2. Estimator substitutions

| # | Pre-specified | Substituted | Why |
|---|---|---|---|
| 2.1 | Feature-tokenised transformer (Gorishniy et al.) as the learner-family control | Multi-layer perceptron (`sklearn.neural_network.MLPClassifier`, 64–32, minority oversampling) | PyTorch was believed unavailable when this row was first run. **RETIRED for the vehicle-claims, Bank Account Fraud and Medicare arms (see 9.1):** `torch` 2.14.0 is importable in this container and `code_run/e22_ftt.py` fits the real architecture on those three. It **stands** for IEEE-CIS and for all three leakage cells of the FTT row, which were not run at all; those cells stay slots and no MLP number is relabelled as an FTT number anywhere. The MLP results keep their own name and their own files. |
| 2.2 | Autoencoder over records (torch) | `sklearn.neural_network.MLPRegressor` fitted to reconstruct its own input through a 16-unit bottleneck | Same reason. It is a genuine undercomplete autoencoder. |
| 2.3 | Data-aware alignment cost as the conformance channel (CONF) | **Implemented and run on the real log; it did not complete over the whole log** (see 4.1), and was then completed over a declared, seeded sample of trace variants (see 9.2). Throughput is still measured on a simulated log. | The estimator is now the one the design names. What is restricted is the population: the CONF, GBDT+$\phi$ and H1(b) cells of the two evaluable flow types are computed by optimal alignment on the aligned subpopulation, whose coverage is stated in the table note. The two-way-match flow yields no detection number (four positives). |
| 2.4 | BINet-style next-event model (TSEQ) | `MLPClassifier` next-event predictor on the previous two events, trace score = mean surprisal | Run on the real log for the `3-way match, invoice after GR` column. It is a genuine next-event sequence model but not BINet's GRU; the TSEQ cells inherit the substitution. |
| 2.5 | Focal loss compared with class weighting, winner chosen on the validation window | class weighting only (`scale_pos_weight`) | Computational budget (see 4). The comparison was not run, so no claim is made about it. |

## 3. Changes of scope forced by the missing log

- **3.1 LeakBench and the E7 gate — superseded 2026-09-20.** The real deviation-type distribution
  now exists, so the gate has a comparator and returns a verdict: **FAIL**. The tolerance was still
  never pre-specified (see 4.3). **All LeakBench results in this run therefore remain properties of
  the benchmark and bear on the dichotomy not at all** — now for a reason backed by data rather
  than for want of a comparator.
- **3.2 H4 value at budget.** The pre-specified pairing is BPI 2019 (leakage) × IEEE-CIS/BAF
  (fraud), paired by day. The fraud side is still unobtainable, and the leakage queue was **not**
  recomputed on the real log within this run's budget, so the H4 numbers are unchanged from
  2026-09-19 and keep the standing described here. The run instead pairs the **real** ULB fraud
  labels and amounts with a **LeakBench-injected** leakage population in the same file, and the
  budget period is one hour rather than one day, because the ULB log spans only two days. Both
  changes are in the result's name and in every sentence that reports it. The H4 numbers are
  therefore a statement about the policy on a semi-synthetic instance, not about a real two-queue
  deployment.
- **3.3 H2 generative model (S3).** S3 fits the legitimate attribute distributions to BPI 2019 and
  samples traces from the model discovered on it. Without the log, the **record** model is a
  Gaussian copula with empirical marginals fitted to the real legitimate records of the ULB file,
  and the **process** model is the four-flow-type purchase-to-pay net documented for BPI 2019 by
  its organisers, written out explicitly rather than discovered. Everything else in S3 is followed:
  the adversary produces the fraudulent records at round 0 inside the density-based plausibility
  set, fraudulent traces enter at the weak-control step with probability π ∈ {0, 0.25, 0.5, 0.75},
  the positive rate is 1%, five seeds, GBDT with and without φ(σ), recall at 1% FPR.
- **3.4 Throughput.** The manuscript reports throughput over the 251,734 BPI cases. The alignment
  figure here is measured on a simulated purchase-to-pay log of the same shape and is projected to
  251,734 cases arithmetically; it is not a measurement on the real log.

## 4. Computational budget

The run shared two CPU cores and 7 GB of RAM with two other analyses under a 100-minute wall-clock
limit. The following reductions were made, all before results were examined:

- ULB detector fitting uses a training subsample of 50,417 rows (all 417 training-window positives
  kept, 50,000 legitimate rows sampled at seed 0) rather than the full 227,845-row training window.
  The test split is untouched and is the full chronological last 20% (56,962 rows, 75 positives).
- LightGBM: 120 trees at learning rate 0.10 (the scaffold's implied default was 300 at 0.05);
  isolation forest: 60 trees; autoencoder: 4 passes; MLP control: 6 passes.
- The adversary grid was planned as the **cross** of the pre-specified 4 × 3 grid — every q at the
  primary κ and every κ at the primary q — rather than the full product, with three seeds at the
  primary cell and one elsewhere. Within the budget only q ∈ {0.01, 0.05, 0.10} at κ = 0.1 were
  reached on Medicare (three seeds at q = 0.05, frozen and retrained); q = 0.25, the κ = 0 and
  κ = 0.5 cells and the whole vehicle-claims arm were not. The "dominant free quantity" slot is
  therefore left open. The labelling delay Δ enters only as the round-to-calendar mapping, so the three
  values of Δ are reported as a mapping and cost no extra runs.
- Hyper-parameters were **not** tuned on a validation window; the settings above were fixed before
  any result was computed and applied unchanged to every dataset.


## 4.1 Alignment-based conformance did not complete (new, 2026-09-20)

The CONF channel is alignment-based conformance (PM4Py A*) against an Inductive-Miner-infrequent
normative model discovered per flow type on the reference set of Section 4.2. It was implemented
(`code_run/e11_conformance.py`) and run, and it **did not complete within the wall-clock budget**.
A* with the state-equation heuristic did not return alignments for a 600-variant sample of the
`3-way match, invoice after GR` flow in nine minutes; reducing to 250 and then 60 variants and
switching to Dijkstra without heuristics did not finish either before the budget expired.
Consequences, all of them slots left standing rather than filled:

- the **CONF** and **GBDT+$\phi$** rows of Table 3 (all three leakage columns);
- **H1(b)**, the primary endpoint of Hypothesis 1, which is CONF minus GBDT+$\phi$ paired per flow
  type — it is therefore **untested in this run**;
- the "record-only shortfall below the conformance channel" half of H1(a);
- the normative model's **fitness and precision** at the primary noise threshold and the leakage
  recall lost at threshold 0.4 (the fitness/precision grid sits behind the alignment block in the
  same script and was never reached);
- **conformance recall across replays** in H3(c), and the precision-loss and recall terms of the
  process-drift sentence.

This is a compute failure, not a data failure. The log is in `data/raw/bpi2019/` and the script is
in `code_run/`; on hardware that can carry the alignment these rows are computable.

## 4.2 Sampling and specification decisions fixed before computing (leakage side, 2026-09-20)

- **The three window operationalisations of "after clearing."** The manuscript requires three and
  fixes one as primary but does not name them. Fixed before any count was taken:
  Window A (narrowest) = outcome activity strictly after the **last** `Clear Invoice`;
  **Window B (PRIMARY)** = strictly after the **first** `Clear Invoice`;
  Window C (widest) = anywhere in the trace, clearing not required. A ⊆ B ⊆ C by construction.
- **Primary Inductive Miner noise threshold** fixed at **0.2** before discovery.
- **Reference set.** Section 4.2 defines it as clean-ending cases "that fall in the modal trace
  variant of their flow type". A single-variant reference set makes the noise threshold inoperative,
  so discovery uses the whole clean-ending reference set and the modal variant and its coverage are
  recorded separately (`results/conf_summary.json`). This departure is on the record.
- **The deviation typology** is the six types of Section 4.2, operationalised as: skipped goods
  receipt (clearing with no `Record Goods Receipt`); invoice before purchase order (first invoice
  event before `Create Purchase Order Item`); duplicate invoice (≥ 2 `Record Invoice Receipt`);
  repeated clearing (≥ 2 `Clear Invoice`); value mismatch (invoice-side cumulative net worth
  exceeding goods-receipt-side by more than 0.5%); atypical role (an event executed by a resource
  performing that activity in fewer than 0.1% of its occurrences). Fixed before computing.
- **Leakage detector evaluation.** Five seeds, 70/30 stratified random split per seed, value-weighted
  recall at a 1% false-positive rate with value = the case's maximum absolute cumulative net worth.
  Record-only features are static case attributes only (value, month of first event, GR-based
  invoice-verification and goods-receipt flags, and code/frequency encodings of vendor, spend area,
  document type, item type and company); no trace information enters them.
- **TSEQ substitution.** The pre-specified next-event model is BINet (a GRU). What was run is an
  `MLPClassifier` next-event predictor on the previous two events, scored as the trace's mean
  surprisal. It is a genuine next-event sequence model but **not** BINet, and the TSEQ row inherits
  this substitution exactly as the FTT row inherits substitution 2.1.
- **Only the `3-way match, invoice after GR` column completed five seeds.** The
  `3-way match, invoice before GR` column (221,010 cases) and the `2-way match` column (1,044 cases,
  4 positives) did not finish inside the budget; their cells stay slots.

## 4.3 The E7 gate tolerance was not pre-specified (new, 2026-09-20)

The manuscript's sentence says the gate's tolerance was "fixed before any outcome was examined".
It was not: `configs/default.yaml` recorded `leakbench_gate.tolerance_jsd` as `not_prespecified` and
no threshold ever existed. A tolerance of **JSD = 0.10** was fixed in this run and written to
`results/e7_gate_tolerance_commitment.json` **before** the distance was computed, but **after** the
seeded results of 2026-09-19 were produced. The sentence has been recast in `main_results.tex` to
say exactly that. The verdict does not turn on the tolerance: the injected and real deviation-type
vocabularies share **no type at all**, so the literal Jensen-Shannon divergence is 1.00, its
maximum, and 0.20 under the coarsest mapping that gives them common support.

## 4.4 H3(c) drift: what was and was not computed

The stationarity endpoint (JSD of the deviation-type mix between halves of the log against a
1,000-draw permutation reference) is computed. The **process-drift curve** is computed as the
monthly JSD of the deviation-type mix against the first third of the log
(`results/h3c_process_drift.csv`), which is **not** the quantity the manuscript's sentence names —
that sentence names the precision a normative model loses by the last month and what recall does.
That needs the conformance model, so those two slots stay open. The monthly series is also
**right-censored**: a case starting in December 2018 has little time to record a post-clearing
outcome before the log ends in January 2019, so the fall in the positive rate over the last three
months (0.38% in October to 0.02% in December) is partly censoring and must not be read as drift.

## 5. Specification decisions fixed before computing

- **Primary adversary setting:** q = 0.05, κ = 0.1, Δ = 90 days. S4 says a primary setting is fixed
  in the pre-analysis plan but does not name it; this triple was fixed before the first adversary
  run and not changed.
- **Value proxy for the vehicle-claims file:** the file carries no monetary amount, so the
  value-weighted metrics use the midpoint of the `VehiclePrice` band (15,000 / 24,500 / 34,500 /
  49,500 / 64,500 / 80,000). Fixed before computing.
- **Medicare split:** by the provider's mean claim start date at the 70th percentile, which is the
  closest available reading of S9's "split by claim date within provider" for a provider-level
  label. Fixed before computing.
- **Loss-given-deviation model for injected leakage:** DUP_EXACT and DUP_NEAR 1.0 of the amount,
  WRONG_PAYEE 0.6, TRANSPOSE 0.5, DOUBLE_FX 0.5, SPLIT_PAY 0.3. S2 specifies the *form* of this
  model against a real log; these numbers are a stated stand-in and every value-at-budget figure
  inherits them.

## 6. Not computable for reasons other than data access

- **H5 (practitioner study)** requires human participants and ethics approval. Every H5 slot stands
  empty. No part of the study was simulated.
- **The narrative-rewriting adversary** requires a language-model agent and a synthetic-narrative
  corpus; it enters no hypothesis test and was not run.
- **Explanation fidelity and stability (S6)** attach to H5 and were not run.


---

## 7. The datasets that arrived on 2026-09-20 (Bank Account Fraud, PaySim), and what their arrival forced

### 7.1 BAF has no monetary amount — a value proxy was defined

The primary metric is **value-weighted** recall at a 1% false-positive rate. Bank Account Fraud is
an application file and records **no** monetary amount: no loss, no exposure, no transaction value.
Three candidate fields were examined before anything was fitted: `income` (a 9-level normalised
band, not money), `intended_balcon_amount` (a scaled, largely negative quantity that is not a
currency amount), and `proposed_credit_limit` (12 distinct values, 190–2,100).

**Decision, fixed before any detector was fitted:** the value proxy is `proposed_credit_limit`, the
credit line the application requests, read as the exposure the institution carries if a fraudulent
application is approved. Its mean is 512 over all applications and 834 over the fraudulent ones.
This follows the precedent already in the paper: the vehicle-claims column is weighted by the
`VehiclePrice` band midpoint, which is also a proxy for exposure and not an observed loss.

It is a proxy, and the table note in `main_results.tex` says so. **Count-based recall is computed
for every BAF cell and reported beside the value-weighted figure** in `RESULTS_REPORT.md` and in
`results/tab_dissociation_fraud_raw_BAF.csv`; the H1(c) contrast is reported on both metrics
(`results/tab_h1c_paired_tests_BAF.csv`) and the sign and significance are the same on both.

### 7.2 BAF: Base variant only

Variants I–V of BAF exist in the author's download but were not extracted. Base is the canonical
variant and is the one used. The manuscript's Table 2 row says "6 variants"; the five
bias-controlled variants are **not** exercised in this run, so no claim about bias or about
variant-to-variant drift is made from BAF.

### 7.3 BAF: split and training subsample

Temporal split, as `splits: {type: temporal}` requires and as the BAF paper itself uses: train on
months 0–5 (794,989 rows, 8,151 positives), test on months 6–7 (205,011 rows, 2,878 positives).
`month` is the split variable and is **not** a feature. For the computational budget the training
set was subsampled to all 8,151 positives plus 120,000 seeded random negatives (128,151 rows);
the test split is used whole. Recorded here, fixed before fitting, unchanged afterwards.

### 7.4 PaySim as the LeakBench host

`configs/default.yaml` lists `paysim` among `datasets:`, and the 19-September run hosted LeakBench
on the ULB test split because PaySim was unobtainable. Both hosts are now reported: the E7 gate
(`results/e7_gate_paysim.json`) and the H4 grid (`results/h4_primary_endpoint_paysim.json`).
Two things about the PaySim host must be stated with the numbers:

- the test split (steps above the 80th percentile) was subsampled uniformly at random with seed 0
  to **303,591 records** for the injection and triage grid — a computational-budget subsample that
  preserves prevalence in expectation, not a change of host;
- the fraud side of PaySim is **trivially separable**: a LightGBM on the standard balance-error
  features reaches a test **PR-AUC of 1.000**. PaySim is a simulator whose fraudulent transfers
  empty the origin account exactly, so the fraud queue is solved by construction. Any H4 number on
  this host is therefore a statement about a host with a perfect fraud scorer and a seeded leakage
  population, and is weaker evidence than the ULB host, not stronger.

### 7.5 A defect found in the 19-September H4 result: injected records formed their own budget periods

`src/leakbench/inject.py` offsets an injected record's timestamp by a uniform fraction of a day
(`c["day"] = r.day + rng.uniform(0, dt)`), so every injected record carries a **fractional** day.
`value_at_budget` groups by `day`. A fractional day matches no base record, so each injected record
became a budget period of its own containing exactly one record, and `k = max(1, round(k_frac*1))
= 1` selected it. The consequence is visible in `results/tab_vab_full_grid.csv`: **`VaB_L = 1.000`
for every policy at every budget** — oracle, both value-ranked queues, fraud-first and the fixed
split alike. The leakage half of the 19-September H4 result is vacuous.

`code_run/e18_h4_period_fix.py` floors `day` onto the host's own period grid after injection and
recomputes the grid (`results/h4_primary_endpoint_periodfix.json`,
`results/tab_vab_pivot_periodfix.csv`). The uncorrected files are left in place. The corrected
numbers are what `main_results.tex` carries; `RESULTS_REPORT.md` reports both and says which slots
moved. This is a defect in the analysis code, not a deviation from the design, and it is recorded
here because it changes published numbers.

### 7.6 Token-based replay substituted for optimal alignment in the CONF channel (H1(b))

Optimal alignment did not complete (4.1). `logs/e11_conf.log` is the evidence: of 600 trace variants
submitted for `3-way match, invoice after GR`, the first ~200 returned in five seconds and the
remainder cost **10–15 seconds each** under both A* with the state-equation heuristic and Dijkstra
without heuristics, projecting to over an hour for one flow type at one noise threshold.

`code_run/e17_conformance_replay.py` substitutes **token-based replay fitness** (Rozinat & van der
Aalst) against the same Inductive-Miner-infrequent model, discovered from the same reference set at
the same primary noise threshold 0.2. The conformance statistic is
`delta_replay = missing tokens + remaining tokens`, computed once per distinct trace variant and
mapped to every case with that variant, so coverage of the evaluation frame is **100%** — there is
no timeout bias.

Sampling frame, fixed before computing (`results/conf_replay_sampling_frame.csv`):
`3-way match, invoice after GR` — all 15,182 cases, 4,228 variants; `3-way match, invoice before GR`
— all 3,396 positives plus 30,000 seeded random negatives of the 221,010-case flow, 3,301 variants;
`2-way match` — all 1,044 cases, 148 variants (4 positives: not evaluable).

**This is a substitution of estimator, not a restriction of sample.** Token replay is not optimal
alignment, and the manuscript's own sentence names "alignment-derived features". The CONF and
GBDT+$\phi$ cells and the H1(b) paired difference therefore **stay slots**, exactly as the FTT cells
stay slots under 2.1, and every replay-based number is reported in `RESULTS_REPORT.md` instead.

**Superseded 2026-09-20 (evening) for the two evaluable flow types.** `code_run/e23_alignments.py`
keeps the specified estimator and restricts the population instead (see 9.2), so the CONF,
GBDT+$\phi$ and H1(b) cells of `3-way match, invoice after GR` and `3-way match, invoice before GR`
are now filled from optimal alignments. The reversal is a change in the estimator, not in the
standard: a declared, seeded variant sample analysed with the named estimator is a restricted
population, whereas token replay is a different estimator, and only the first of those two can
fill a slot that names the estimator. Every token-replay number remains reported in
`RESULTS_REPORT.md` and enters no slot.

### 7.7 The H1(a) leakage cells were diagnosed, not withdrawn

`code_run/e15_h1a_diagnosis.py` re-ran the leakage detectors under the identical protocol and added,
for every cell, the count-based recall at the same threshold, the Spearman correlation between score
and case value, and two reference detectors: a ranker that uses **case value alone** and a random
ranker. The value-weighted figures stand as correct computations of the metric the table names, and
the table note now carries the value-only null beside them. The diagnosis is in `RESULTS_REPORT.md`.

## 8. IEEE-CIS arrived on 2026-09-20 (third addendum), and what its arrival forced

IEEE-CIS was listed in section 1 as unobtainable. It was fetched on the author's machine and
installed at `data/raw/ieee_cis/` on 2026-09-20; the counts, hashes and schema check are in
`DATA_PROVENANCE.md`. Section 1's entry for IEEE-CIS is superseded. The endpoints it unblocked are
the IEEE-CIS column of Table 3 (`code_run/e20_ieee_dissociation.py`) and the restricted adaptive
adversary of H3 (`code_run/e21_ieee_adversary.py`). Everything below was fixed before any IEEE-CIS
number was computed.

### 8.1 Temporal split on `TransactionDT`, which is a delta and not a timestamp

`TransactionDT` is documented as a time delta in seconds from an unstated reference. It is
therefore used **only as an ordering variable**: rows are sorted by it and cut at its 75th
percentile (11,246,620 s, day 129.2 of the file's 182.0-day span), giving 442,905 training rows
and 147,635 test rows. This is the same 75/25 temporal proportion as the BAF arm (months 0–5 /
6–7) and satisfies the supplement's rule that "IEEE-CIS and Bank Account Fraud are split on their
time fields". `TransactionDT` is **not** a feature, exactly as `month` is not a feature in the BAF
arm. The positive rate is 3.514% in the training window and 3.454% in the test window, so the split
does not concentrate the positives.

### 8.2 The 75.6% of transactions with no identity record are kept, and missingness is imputed once

`train_identity.csv` covers 144,233 of the 590,540 transactions (24.4239%). Dropping the remaining
446,307 would discard three quarters of the dataset and change the population the column describes,
so the join is a **left** join: the unmatched rows keep their 40 identity columns missing and carry
a binary `has_identity` feature.

Missing entries — 115,523,073 cells, 45.3% of the 590,540 x 432 matrix — are replaced by the
**training-window median** of their column. No column is entirely missing within the training
window and none is constant there, so no column was dropped on those grounds. The same imputed
matrix is given to all five detectors. This is a deliberate departure from what LightGBM would do
by default: LightGBM handles `NaN` natively and the isolation forest, the autoencoder and the
neural control do not, so leaving the missingness in place would have given the supervised detector
strictly more information than the unsupervised ones in a comparison whose whole content is which
detector wins. Missingness is not thereby discarded: `has_identity` retains the one missingness
pattern that is interpretable.

### 8.3 Training subsample

All 15,563 training-window positives are kept; the negatives are subsampled to 120,000 with seed 0,
giving 135,563 training rows. This is the identical computational-budget rule used in the BAF arm
(7.3). The test split is used whole (147,635 rows, 5,100 positives).

### 8.4 IEEE-CIS is the only fraud column whose value weighting is an observed amount

This is a correction to the table, not a deviation from the design. `TransactionAmt` is a real
monetary field (median 68.769, max 31,937.39), so the IEEE-CIS column's value-weighted recall is a
value metric on observed amounts. The Bank Account Fraud column is weighted by
`proposed_credit_limit`, an exposure **proxy** (7.1); the vehicle column by the midpoint of a coded
price band (7.2); the Medicare column by total reimbursement, which is an amount but at provider
level. The note to Table 3 now says which columns are proxies and which is not. Count-based recall
is reported beside every IEEE-CIS cell in `RESULTS_REPORT.md`.

### 8.5 The restricted adversary: the mutable set, and two protocol departures

The supplement states "IEEE-CIS is partly anonymised, so the mutable set is restricted
accordingly", and the claim-by-evidence table lists H3 fraud durability on "IEEE-CIS (restricted)".
The mutable set was fixed, before any adversary run, to the **ten** fields that are both
un-anonymised and under the transacting party's control:

    TransactionAmt, dist1, card4, card6, P_emaildomain, R_emaildomain,
    DeviceType, DeviceInfo, id_30 (operating system), id_31 (browser)

Held immutable: `V1`–`V339` and `id_01`–`id_29` (anonymised engineered features whose semantics
Vesta does not release), `C1`–`C14` and `D1`–`D15` (counts and time deltas the acquirer computes),
`addr1`/`addr2` and `card1`/`card2`/`card3`/`card5` (instrument and billing identity), `M1`–`M9`
(processor match flags). Ten mutable coordinates is the same count as the vehicle arm's.

Two departures from `e4_durability.py`'s implementation, both forced by the dataset's shape:

1. **The plausibility copula is fitted to the ten mutable coordinates only.** On the vehicle (31
   coordinates) and Medicare (13) arms it spans every coordinate. Over IEEE-CIS's 432 coordinates a
   Gaussian copula estimated from 8,000 legitimate records is dominated by estimation error
   (n/d ≈ 18), and the constraint is in any case only ever evaluated on coordinates the adversary
   moves. The consequence is that the plausibility set is a **marginal** density constraint on the
   mutable block rather than a joint one over all fields, which is a weaker constraint; a weaker
   constraint makes the adversary stronger, so the IEEE-CIS decay curve is, if anything, an
   upper bound on the rate of decay rather than a conservative one.
2. **The six coded-categorical mutable coordinates are snapped to the coded lattice.** After each
   coordinate move, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`,
   `id_30` and `id_31` are rounded and clipped to the code range observed in training, so the
   adversary cannot occupy a value the encoding does not admit. The vehicle arm moved its coded
   fields continuously; this is stricter, not looser.

The grid, the number of rounds, the primary cell and the reduced cross are those of S4 and of
`e4_durability.py` unchanged: q ∈ {0.01, 0.05, 0.10, 0.25}, κ ∈ {0, 0.1, 0.5}, Δ ∈ {30, 90, 180}
days, T = 10, primary (q = 0.05, κ = 0.1, Δ = 90), three seeds at the primary cell and one
elsewhere, frozen and retrained detectors.

### 8.6 What IEEE-CIS makes computable and was still **not** computed

Reported here so that the gap is a budget decision on the record rather than a silence:

- **The pre-specified H4 triage pairing, "BPI 2019 (leakage) x IEEE-CIS/BAF (fraud), paired by
  day".** Both halves now exist. It was not computed within this run's 75-minute budget, and the
  value-at-budget figures in the manuscript remain those of the ULB + LeakBench host, with the
  PaySim host reported beside them. The sentence that named the host has been recast: it previously
  said the pairing "was not available to this run", which is no longer true, and now says the
  pairing was not computed. This is the single largest piece of work IEEE-CIS unblocks and does not
  close.
- **The E7 injection gate on an IEEE-CIS host.** The claim-by-evidence table names IEEE-CIS, BAF
  and PaySim as the injection hosts; the gate was run on a ULB host and a PaySim host. The
  IEEE-CIS host was not added. Not computed, budget.
- **The feature-tokenised transformer.** PyTorch was not installed and no FTT was run, on IEEE-CIS
  or anywhere else. The neural control in the IEEE-CIS column is the same MLP substitute as
  everywhere else (2.1), so the **FTT cell of the IEEE-CIS column stays a slot** and the sentence
  "the feature-tokenised transformer sits … the tree ensemble" stays open. The MLP's figures are in
  `RESULTS_REPORT.md`.
- **The CONF cell of the IEEE-CIS column.** There is no process log behind IEEE-CIS, so there is no
  conformance channel to run on it, exactly as for BAF and the vehicle file. The cell stays a slot.
- **H5.** The practitioner study draws 16 of its 32 cases from IEEE-CIS. It needs participants and
  ethics approval; every H5 slot stays open.
- **The five BAF bias variants (Variant I–V)** remain unextracted (7.2). No variant-level claim is
  made anywhere; the author's machine holds them and they can be fetched if the design is read as
  requiring one.

### 8.7 A defect in the retraining arm of the adversary harness (found 2026-09-20, **not fixed**)

`code_run/e4_durability.py` appends the round-$t$ adversarial records to the training set
(`tr_X = [tr_X; Xf]`), refits the detector, and then at round $t+1$ scores **the same `Xf`** before
the adversary moves again. The retrained detector is therefore evaluated in-sample on the exact
adversarial records it was just trained on. On IEEE-CIS this produces $R_1 = 96.7\%$ under
retraining against a clean-data $R_0 = 31.5\%$; the same mechanism drives the Medicare retrained
path.

`e21_ieee_adversary.py` reuses the protocol **unchanged**, so that the two datasets are comparable,
and inherits the defect. It is recorded rather than repaired because the two retraining slots
already filled in the manuscript — "path shape under retraining" and "retrained path against the
frozen path", both from the Medicare run — would change, and changing an already-reported result
is an author decision. **No IEEE-CIS retraining number was written into the manuscript.** The
frozen-detector arm, which carries the pre-registered primary endpoint $V_T/V_0$, is unaffected:
the frozen detector is never refitted.

### 8.8 The IEEE-CIS adversary grid completed; `dominant free quantity` still stays a slot

The reduced cross of S4 completed on IEEE-CIS: all 16 (q, κ, retrain, seed) runs reached round 10.
Spread in last-round value-weighted recall is SD 0.0455 across q, SD 0.0469 across κ and 0 across
Δ. The slot `dominant free quantity` is nevertheless **not** filled: the sentence it sits in
reports the Medicare run (where κ was never varied), so an IEEE-CIS answer would be a different
dataset's quantity in that sentence; and on IEEE-CIS the two candidates differ by 0.0014, which is
not a dominance. `code_run/e21b_ieee_adv_summary.py` builds the summary files from whatever cells
have completed and records which they are, so the same endpoint is recoverable from a truncated
run.


## 9. The run of 2026-09-20 (evening): the real transformer, optimal alignments, the H4 pairing, BPI 2017

The two addenda written by the agents that produced these results, `DEVIATIONS_ADDENDUM_A.md` and
`DEVIATIONS_ADDENDUM_B.md`, are kept beside this file unchanged and are merged here verbatim so
that this log remains the single list of departures from the pre-specified design. Section 9.x is
addendum A (the feature-tokenised transformer and alignment-based conformance, scripts `e22`,
`e23`); section 10.x is addendum B (the H4 pairing, the kappa cells, the flow-2 column, BPI 2017
and the third E7 host, scripts `e24`–`e28`). Where they retire or supersede an entry above, the
entry above now says so.

One departure in section 9 was introduced in *this* consolidation pass and is not in either
addendum: **the vehicle-claims arm of the transformer was re-run to 40 epochs**. Addendum A.1.3-a
records that arm as undertrained at 8 epochs with the training loss still falling, so the arm was
re-run on uncontended CPU with everything else held fixed — same split, same five seeds, same
architecture, same metric. The 8-epoch rows are kept for provenance in
`results/ftt_raw_prebudgetfix.csv` and `results/ftt_config_prebudgetfix.json`; the epoch count is
`VEHICLE_EPOCHS` in `code_run/e22_ftt.py`, default 40, and `run_all.sh` passes it. The Bank Account
Fraud arm's loss was also still falling at its 2-epoch budget and was **not** re-run — that arm
costs about thirteen minutes per seed — so its cell remains a lower bound on what this
architecture reaches on those data, and A.1.3-a's qualification stands for it.

**Scope.** This addendum covers only the two blocks of work run on 2026-09-20 under
`code_run/e22_ftt.py` and `code_run/e23_alignments.py`, whose outputs are `results/ftt_*` and
`results/conf_align_*`. It adds to `DEVIATIONS.md`; it replaces nothing in it. Where it *retires*
a deviation already logged there, it says so and names it, but it does not edit that file.

Everything below is a real computed quantity or a statement that a quantity was not computed.
No near-neighbour quantity is reported under another quantity's name.


### A.1 Substitution 2.1 (sklearn MLP in place of the FTT) is retired where marked, and only there

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

#### A.1.1 What was implemented

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

#### A.1.2 Protocol held fixed

Identical to `e1_dissociation.py`, `e14_baf_dissociation.py` and `e20_ieee_dissociation.py`: the
same splits (read from those scripts, not re-invented), the same five seeds `[0,1,2,3,4]`,
value-weighted recall at a 1% false-positive rate with count-based recall and PR-AUC beside it.
Class imbalance is handled by `pos_weight = n_neg/n_pos` in the BCE loss, the loss-side equivalent
of the `scale_pos_weight` LightGBM is given in `common.fit_gbdt`.

#### A.1.3 New deviations this block introduces

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


### A.2 Alignment-based conformance: a declared variant sample, not an estimator substitution

`DEVIATIONS.md` 7.6 logs token-based replay as a **substitution** for optimal alignment, and that is
why the CONF, GBDT+φ and H1(b) cells stayed open: token replay is a different estimator from the
one §4.2 and Supplementary S2 specify.

`code_run/e23_alignments.py` **keeps the estimator** — PM4Py A* optimal alignment, unit cost for a
log move and for a model move, zero for a synchronous move — against **the same** normative model:
the same reference set (cases of the flow type whose trace ends in `Clear Invoice` with no
correction activity), the same Inductive Miner (infrequent), the same primary noise threshold 0.2,
discovered exactly as `e11_conformance.py` and `e17_conformance_replay.py` discover it. What is
restricted is the **population**, not the estimator.

#### A.2.1 Sample design (this is the deviation)

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

#### A.2.2 What the sample costs, and the bias it carries

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

#### A.2.3 Detection protocol

Identical to `e17_conformance_replay.py`, so the alignment estimator and the replay estimator are
directly comparable:

| detector | input |
|---|---|
| CONF (alignments) | the alignment cost δ |
| GBDT+φ (alignments) | record features **plus** φ(σ) = (alignment cost, number of log moves, number of model moves, trace length, six typed-deviation indicators) |
| GBDT (records), IF, AE | record features only, exactly as in `e17` |

Five seeds, stratified 70/30 split per seed, value-weighted recall at a 1% false-positive rate with
count-based recall beside it. H1(b) is the paired difference CONF − GBDT+φ.

#### A.2.4 What is still not computed on the leakage side

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

#### A.2.5 Comparability with the existing leakage cells

The leakage cells already in `main_results.tex` (Rules, IF, AE, TSEQ, GBDT records for
3-way-after-GR) come from `tab_dissociation_leakage_raw.csv`, whose evaluation set is the 30% test
split of the **whole** flow (n = 4,555, 205 positives). The alignment cells come from the 30% test
split of the **evaluated subpopulation**. These are different populations. Putting a CONF cell from
this run beside an IF cell from that run in one table column is not a like-for-like comparison, and
the table note should say so, or the whole column should be rebuilt from
`conf_align_detection_raw.csv`, which contains IF, AE and GBDT (records) on the alignment
population for exactly that purpose.

#### A.2.6 Ablation: is GBDT+φ winning on alignment features or on the typed deviations?

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

---

## 10. Addendum B (verbatim): the run of 2026-09-20 (evening), block B

This addendum is additive. It records the departures from the manuscript's pre-specified design
introduced by five analyses run on 2026-09-20 in the evening, and nothing else; `DEVIATIONS.md`
and everything it records stand unchanged. Scripts: `code_run/e24_h4_pairing.py`,
`e25_kappa_cells.py`, `e26_flow2_column.py`, `e27_bpi2017.py`, `e28_e7_ieee.py`. Results:
`results/h4pair_*`, `kappa_*`, `flow2_*`, `bpi2017_*`, `e7ieee_*`. Slot mapping:
`results/handoff_B.json`.

As in `DEVIATIONS.md`: these analyses were executed at the authors' decision, before OSF
registration of the pre-analysis plan; the manuscript's registration language must be revised
accordingly.


### B.1 H4 on the pre-specified pairing (BPI 2019 × IEEE-CIS)

`e24_h4_pairing.py`. This is the host the design names. Both halves were obtained on 2026-09-20,
so the pairing could be built for the first time. It supersedes nothing automatically: the ULB
host (`h4_primary_endpoint_periodfix.json`) and the PaySim host
(`h4_primary_endpoint_paysim_periodfix.json`) remain computed and remain reported.

**B.1.1 The two files are not contemporaneous, so "paired by day" is pairing by day *rank*.**
IEEE-CIS publishes `TransactionDT` as a delta in seconds with no calendar origin; BPI 2019 runs
from January 2018 to January 2019. A calendar join is impossible. The fraud test window's 53
distinct days are the budget periods; the leakage test window's 117 days are distributed onto that
grid in time order in proportion (`floor(r · 53 / 117)`). Every record on both sides is used and
every period carries records of both intents (period sizes 2,099 / 4,118 / 6,869 for min / median
/ max). This is a construction choice; it is not in the manuscript, and it is the only pairing the
two files admit.

**B.1.2 Each record is scored by the channel that applies to it.** A card transaction carries no
purchase-order trace and a purchase-order item carries no card features, so p_L = 0 on IEEE-CIS
records and p_F = 0 on BPI 2019 records. A consequence worth stating because it is exact and not
an artefact of estimation: the fraud-first policy captures **zero** leakage value at every budget,
because it ranks on p_F alone and every leakage record has p_F = 0.

**B.1.3 The leakage score is a supervised record-only learner, not the conformance channel.**
p_F on the fraud side is the supervised GBDT of `e20`. For symmetry p_L is a supervised GBDT on
the record-only case attributes of `e12`, fitted on the leakage training window only and applied
out of sample. The ULB-host H4 used a rule detector as the leakage scorer because that host had no
labels. The two H4 instances therefore do not use the same leakage scorer, and the levels are not
comparable across hosts — only the *ordering of policies* is.

**B.1.4 The two sides are in different currencies and are read at face value.** IEEE-CIS is USD,
BPI 2019 is EUR. The policies rank on expected value, so the relative scale of the two sides is
part of the pairing. The primary run applies no conversion. A secondary run, declared in the
script before either was computed, rescales leakage value so that the total leakage value
available equals the total fraud value available (factor 0.4985); both are reported. The
conclusion does not turn on it.

**B.1.5 The temporal split truncates the leakage label's follow-up.** Outcome-derived labels
(window B) depend on what happens after a case opens, so cases that open late in the log have less
follow-up. The leakage positive rate is 2.04% in the training window and **0.375%** in the test
window, and the out-of-sample PR-AUC of the leakage learner is **0.0123** against a base rate of
0.00375. The leakage queue in this pairing is therefore scored by a weak model on a
label-attenuated window. That is a property of a temporal split on outcome labels, it is reported
with the result, and it bounds how much weight the leakage half of the H4 number can carry.

**B.1.6 The period defect of `DEVIATIONS.md` 7.5 cannot recur here and was checked anyway.**
Nothing is injected; both sides are real records floored onto an integer period grid. Singleton
periods: zero. VaB_L is not identically one: over the whole grid the two-queue policy's VaB_L runs
from 0.636 to 0.917 and the single queue's from 0.636 to 0.920. The oracle's VaB_L runs from 0.969
to 1.000 and reaches exactly 1.000 at the largest budgets, which is what an oracle does when the
positives are few and concentrated in value; it is not the defect.

### B.2 The kappa cells of the H3 adversary grid

`e25_kappa_cells.py`.

**B.2.1 Frozen arm only.** `e4_durability.py` retrains the detector on the adversary's own
perturbed records and then scores those same records (`DEVIATIONS.md` 8.7). Nothing in this
analysis is computed from the retrained arm, including the "dominant free quantity" decision. The
retraining switch produces a larger spread than any of the three free quantities on the IEEE-CIS
grid, and that comparison is not used here for the same reason.

**B.2.2 Three seeds at every cell, not one.** `e4` ran three seeds at the primary cell and one
elsewhere. Deciding which free quantity dominates needs a noise reference at every cell, so all
six cells were run at three seeds. The endpoint, the estimator and the grid are unchanged.

**B.2.3 The labelling delay Δ contributes an exact zero, by construction.** A round is one
labelling delay, so Δ maps rounds to calendar time and cannot move R_t as a function of round
index. Its spread is reported as 0.0 with that reason, not as an unmeasured quantity. Any claim
that one of *three* free quantities dominates therefore rests on a comparison of two that vary and
one that cannot.

**B.2.4 The grid is a cross, not a full factorial** (`DEVIATIONS.md`, e4's own deviation): every q
at the primary κ and every κ at the primary q. The q arm has four points and the κ arm three, so
their standard deviations are not computed over equal numbers of points; the ranges are reported
beside them and give the same ordering.

### B.3 The `3-way match, invoice before GR` leakage column

`e26_flow2_column.py`.

**B.3.1 This is the population, and the earlier run was a negative sample.** The 2026-09-20
midday run (`e17`) evaluated this flow on 3,396 positives plus a seeded sample of 30,000 of the
217,614 negatives. Value-weighted recall at a fixed false-positive rate is a quantile of the
*negative* score distribution, so it is not invariant to that sample. All 221,010 cases are used
here. The two are compared in `results/flow2_population_vs_sample.csv`; the sampled frame was
mildly optimistic for the supervised arms (GBDT 51.3% → 47.8%, GBDT+φ 80.4% → 76.8%).

**B.3.2 CONF and GBDT+φ are token-based replay, not alignments** — `DEVIATIONS.md` 7.6,
unchanged. Replay is computed once per distinct trace variant of the full population (7,832
variants) and mapped to every case, so coverage of the evaluation frame is 100%. These two rows
are **not** proposed for the manuscript, for the same reason the earlier replay rows were not: the
sentence names alignment-derived features and token replay is a different conformance estimator.

**B.3.3 The value-only null and the random floor are scored in the same runs.** They are not
detectors of the design; they are controls for the concentration of positive value documented in
`RESULTS_REPORT.md` 0.2, and they are what makes the value-weighted numbers in this column
readable. On this flow the supervised record-only learner reaches 47.8% value-weighted recall and
a ranker that uses **case value and nothing else** reaches 47.7%: the difference is +0.09
percentage points, p = 0.87, and 3 of 5 seeds. On count-based recall the supervised learner is
16.2% against the null's 5.9%, p = 2 × 10⁻⁹, 5 of 5 seeds. The column's value-weighted cells are
correct computations of the quantity Table 3 names and they do not, on their own, demonstrate
detection.

### B.4 BPI Challenge 2017 — the process-generality check

`e27_bpi2017.py`. The manuscript assigns this log one job and excludes it from every other
result. Only that job is computed: the published counts are verified, a normative model is
discovered over the same noise grid with the same primary threshold, its fitness and precision are
reported, and the cost of optimal alignment is measured the way it was measured for BPI 2019.

**B.4.1 The reference-set rule was corrected once, before any outcome was computed.** The BPI 2019
channel takes as reference the cases that reach the normal terminal activity with no correction
activity. Transcribed to the loan process, the first rule required A_Pending and excluded
A_Cancelled, A_Denied, O_Cancelled, O_Refused and O_Returned; it returns **zero** reference cases,
because a BPI 2017 application carries several offers and offer-level returns and cancellations
are part of the normal process. The rule was corrected to the application-level outcome alone
(reaches A_Pending, contains neither A_Cancelled nor A_Denied) before any fitness or precision
number existed. The empty run is in `logs/e27_bpi2017.log` and the correction is in the script.

**B.4.2 Activity labels ignore `lifecycle:transition`**, as `e9`/`e11` do for BPI 2019. The
variant count under a complete-events-only reading is reported beside the primary count as a
sensitivity (15,930 against 5,623).

**B.4.3 Only the primary noise threshold was run.** One pass over this log (discovery + token
replay fitness over all 15,930 variants + ETC precision over the 2,000 most frequent) costs about
256 s, and the four-point noise grid does not fit the remaining budget. The grid is ordered
primary-first and stops when the budget is spent, so `results/bpi2017_noise_grid.csv` carries
noise = 0.2 and three "time budget" rows. A completed noise = 0.0 pass from an earlier run of the
same script is kept in `logs/e27_bpi2017_noise0_run.log` (fitness 0.9916, precision 0.1280).

**B.4.4 Alignment is measured twice, on the frequent head and on a uniform sample.** The BPI 2019
alignment evidence in `logs/e11_conf.log` is a head-of-distribution measurement that blew up in
the tail, so measuring only the head here would not be a comparable answer. Both are reported:
the 400 most frequent variants (all completed, median 0.055 s each) and a uniform sample of 300
of the 15,930 variants (all completed, median 0.097 s, p90 0.18 s, max 1.25 s). Neither hit the
150 s budget. The `cost` figure PM4Py returns is its raw cost scale (10,000 per deviating move),
not a move count; the interpretable quantity is the alignment fitness beside it.

### B.5 The E7 injection gate on an IEEE-CIS host

`e28_e7_ieee.py`. No deviation from the gate's construction: same seeded type mix, same rates,
same hard and easy variants, same three seeds, same tolerance commitment file
(`e7_gate_tolerance_commitment.json`, whose own deviation — the tolerance was fixed after the
seeded results existed, because the configuration left it unset — stands and is unchanged), same
real comparator. Two host-construction choices are new and are recorded here: `entity_id` is
`card1` and `counterparty_id` is the `P_emaildomain` code, because IEEE-CIS has no payer or payee
identifier and these are the nearest available; and the host is the cached temporal test split of
`e20`, not a fresh split.

The three hosts the design names have now all been run and the gate fails on all three. The
literal distance is 1.000 on every host because the seeded vocabulary and the BPI 2019 vocabulary
are disjoint; the coarse-mapping distance varies only in the third decimal (ULB 0.196, PaySim
0.192, IEEE-CIS 0.193). The gate's verdict is a property of LeakBench and of the real log, not of
the host.

### D8.9 Stale transformer rows in `results/`, found 2026-09-21

The feature-tokenised transformer was first run as a `scikit-learn` MLP substitute (2.1). The
substitution was later retired and the real architecture was trained on three files. Four sets of
outputs still carry the MLP-era numbers under the name `FTT`: the raw and summary
`tab_dissociation_fraud_*` files for Bank Account Fraud, vehicle claims, Medicare and IEEE-CIS,
and the `tab_h1c_paired_tests*` files derived from them.

Nothing in `main_results.tex` comes from those rows. Table 3's FTT cells are filled from
`ftt_summary.csv`, which holds the current values, and `check_manuscript.py` verifies that every
filled slot traces to the file it names. The rows matter anyway, because one of them reports a
value-weighted recall of 0.2309 for an IEEE-CIS transformer arm the manuscript states was never
fitted. The MLP substitute did run on that host, under the `FTT` label, before the substitution
was retired.

`results/README_SUPERSEDED.md` lists every stale value beside its current one.
`results/ftt_vs_gbdt_paired_tests.csv` is the current paired comparison, computed on 21 September
from `ftt_raw.csv` against the GBDT rows of the same raw files, by seed. It is what the recast
sentence in Section 4.2 reports.

### D8.10 Two result files rewritten as superseded records, 2026-09-21

`h3_leakage_side.json` and `h1_leakage_real_log.json` both recorded their analyses as "not
computed" because BPI Challenge 2019 could not be obtained. The log was obtained on 20 September
and the analyses were run. Both files now carry `status: SUPERSEDED`, their original text, and a
map from each blocked item to the file that now holds it. Neither was ever read by
`fill_slots.py`.
