# Deviations addendum B — run of 2026-09-20 (evening), block B

This addendum is additive. It records the departures from the manuscript's pre-specified design
introduced by five analyses run on 2026-09-20 in the evening, and nothing else; `DEVIATIONS.md`
and everything it records stand unchanged. Scripts: `code_run/e24_h4_pairing.py`,
`e25_kappa_cells.py`, `e26_flow2_column.py`, `e27_bpi2017.py`, `e28_e7_ieee.py`. Results:
`results/h4pair_*`, `kappa_*`, `flow2_*`, `bpi2017_*`, `e7ieee_*`. Slot mapping:
`results/handoff_B.json`.

As in `DEVIATIONS.md`: these analyses were executed at the authors' decision, before OSF
registration of the pre-analysis plan; the manuscript's registration language must be revised
accordingly.

---

## B.1 H4 on the pre-specified pairing (BPI 2019 × IEEE-CIS)

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

## B.2 The kappa cells of the H3 adversary grid

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

## B.3 The `3-way match, invoice before GR` leakage column

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

## B.4 BPI Challenge 2017 — the process-generality check

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

## B.5 The E7 injection gate on an IEEE-CIS host

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
