# Data provenance

Retrieval date: **2026-09-19** for the fraud files, **2026-09-20** for BPI Challenge 2019, times UTC. All downloads by `curl` through the analysis environment's HTTPS
proxy from `raw.githubusercontent.com` only. Every file below was checked against the row and
positive counts the manuscript's Table 2 states and against the dataset's published schema; a
mirror whose counts did not match would not have been used.

## Obtained

| File (`data/raw/…`) | Bytes | SHA-256 | Rows | Cols | Manuscript Table 2 | Verified |
|---|---|---|---|---|---|---|
| `ulb/creditcard.csv` | 102,634,230 | `33a178be65174943a07d1736acf5898f6c8e1fa80950dbbe598c17eda968ec59` | 284,807 | 31 | 284,807 rows / 492 fraud | rows **exact**; `Class` sum = **492**, exact; columns `Time, V1…V28, Amount, Class` as published |
| `carclaims/fraud_oracle.csv` | 3,618,564 | `8b6aa59764ef4f8b058598d3e8f325ef3623f52f4b1cd4ef946baebb5ccfa9a6` | 15,420 | 33 | 15,420 rows / 923 fraud | rows **exact**; `FraudFound_P` sum = **923**, exact; 33 coded fields, no narrative column, as S9 states |
| `medicare/Train-1542865627584.csv` | 87,094 | `09315a0e0c262ddfb91df15df76b196a80de2ff1001fad46b362b562df2e7e43` | 5,410 | 2 | 5,410 providers / 506 flagged | providers **exact**; `PotentialFraud == "Yes"` = **506**, exact |
| `medicare/Train_Beneficiarydata-1542865627584.csv` | 11,444,821 | `1fb7fb14efeaae04b970bfeb3176cdf938c755748217377109081d6dae01ecfd` | 138,556 | 25 | beneficiary file | present |
| `medicare/Train_Inpatientdata-1542865627584.csv` | 8,572,555 | `395bc2d01a102bf44a26ba4b36c63e7086943eaf2d5d1489c776bf47d1613ac0` | 40,474 | 30 | inpatient file | present |
| `medicare/Train_Outpatientdata-1542865627584.csv` | 77,391,090 | `f532ed7e25f50d6abf3ba89ceeb2a23e2a352792f90d086b13999a561fd23889` | 517,737 | 27 | outpatient file | present |

**Claim-level count for the Medicare file** (the `\authornote` in S9 asks for it): 40,474 inpatient +
517,737 outpatient = **558,211 claims** over 5,410 providers and 138,556 beneficiaries.

### Mirrors used and the canonical sources they mirror

- **ULB credit-card fraud** — mirror `github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection`,
  raw path `master/creditcard.csv`. Canonical source: Machine Learning Group, Université Libre de
  Bruxelles, distributed via Kaggle `mlg-ulb/creditcardfraud` (Dal Pozzolo et al. 2015). Kaggle is
  blocked at the egress gateway.
- **Vehicle insurance claims (`carclaims` / `fraud_oracle.csv`)** — mirror
  `github.com/Shritik12345/-vehicle-insurance-fraud-detection`, raw path `main/fraud_oracle.csv`.
  Canonical source: the 1994–96 `carclaims` file of Phua et al. (2004), distributed via Kaggle
  `khusheekapoor/vehicle-insurance-fraud-detection`.
- **Medicare provider fraud** — mirror `github.com/sagnikgh1899/FraudDetection`, raw paths
  `main/FraudDetection/data/Train*-1542865627584.csv`. Canonical source: Kaggle
  `rohitrox/healthcare-provider-fraud-detection-analysis`. The `-1542865627584` suffix is the
  original release's filename stamp and is unchanged in the mirror.

Commit SHAs were not recorded: `api.github.com` answers 403 from the analysis environment and the raw endpoint
was fetched by branch (`master` / `main`) on 2026-09-19. The SHA-256 of each downloaded file above
pins the bytes actually used.

## Not obtained on 2026-09-19, with the search that was made

**The BPI Challenge 2019 row below is superseded**: the log was obtained on 2026-09-20 and is
documented in the section that follows this table. The row is kept as the record of the search.


| Dataset | Needed for | Outcome |
|---|---|---|
| **BPI Challenge 2019** (1,595,923 events / 251,734 cases) | every real-log leakage result: H1(a), H1(b), the leakage population (Table S5), the E7 gate, H3(c) stationarity and drift, the leakage queue in H4, the leakage half of the H5 case set | **not obtained.** Canonical host `data.4tu.nl` (DOI `10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1`) is blocked at the egress gateway (403 on CONNECT). GitHub was searched seriously: WebSearch over five phrasings; ten candidate repositories inspected (`Adeleet/BPIC2019`, `mrsonuk/BPI_Challenge_2019`, `Vincentxyz/BPI_Challenge_2019`, `minanayernia/BPI-Challenge-2019-`, `VijayV28/decoupled-trace-clustering`, `mpcristescu/bpi2019-robust-bottleneck-prioritization`, `sakce/` and `aleksa007/Process-Mining-Business-Intelligence-Challenge`, `isa-group/bpi-challenge-performance-analysis`, `vivek4033/p2p-exception-agent`, `DanieleBocchino/Business-Information-System`, `glenngru/Stacked-Event-Timelines`, `ANILMUDAVATH/Purchase-to-Pay-…`); ~140 raw-endpoint probes across `main`/`master` for `.xes`, `.xes.gz`, `.csv` and `.parquet` under root, `data/`, `data/raw/` and `dataset/`. Every candidate documents the log as a **local download**: `VijayV28/decoupled-trace-clustering/data/README.md` states verbatim "The BPI Challenge 2019 event log is ~695 MB, which exceeds GitHub's file-size limit, so it is not part of this repository", and `mpcristescu/…` keeps only `DATASET_LINKS.md`. No LFS or release-asset mirror was found. `codeload.github.com` and repository browsing on `github.com` are also blocked for repositories not enabled for this session. |
| **BPI Challenge 2017** | process-generality check only | **not obtained** (same host, same block). |
| **IEEE-CIS** (590,540 rows) | the fraud columns of the dissociation table, the restricted adversary, the H5 fraud case set | **not obtained.** Kaggle competition data; Kaggle is blocked; no GitHub mirror of `train_transaction.csv` (≈ 650 MB) was found. |
| **Bank Account Fraud** (6 variants, 1 M each) | fraud, drift, adversary | **not obtained.** `github.com/feedzai/bank-account-fraud` is reachable and its README is readable, but the repository contains **no data**: it points to Kaggle (`sgpjesus/bank-account-fraud-dataset-neurips-2022`), which is blocked. |
| **PaySim** (6,362,620 rows) | the LeakBench host named in `configs/default.yaml` | **not obtained.** Four candidate mirrors probed for `PS_20174392719_1491204439457_log.csv` and `paysim.csv` under root, `data/` and `data/raw/`; none carries the file (≈ 470 MB). LeakBench was therefore hosted on the ULB test split instead — a substitution recorded in `DEVIATIONS.md`. |


## Obtained 2026-09-20 (addendum): BPI Challenge 2019

| File (`data/raw/bpi2019/`) | Bytes | SHA-256 | Cases | Events | Manuscript Table 2 | Verified |
|---|---|---|---|---|---|---|
| `BPI_Challenge_2019.xes.gz` | 28,309,937 | uncompressed XES `af63bc687fc4152f2123b05c3af7772b37ef3fce2d3f67f812666c9e356baae7` | 251,734 | 1,595,923 | 251,734 cases / 1,595,923 events | **exact on both counts** |
| `bpi2019_events.csv` | 442,902,464 | derived (see below) | 251,734 | 1,595,923 | — | one row per event, 21 columns, trace attributes prefixed `case:` |

Source: data.4tu.nl, DOI `10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1`, CC BY 4.0, retrieved
2026-09-20 through a browser on the author's own machine (the host is blocked at the analysis environment's
egress gateway). `bpi2019_events.csv` was produced from the XES by `code_run/xes_to_csv.py`
(streaming `iterparse`; a DOM parse does not fit in 7 GB).

**Independent verification in this run.** `code_run/e9_leakage_prep.py` re-read the flat file in
one chunked pass and counted **1,595,923 events over 251,734 cases**, matching Table 2 exactly, and
enumerated **42 distinct activity labels** (`results/leakage_activity_census.csv`). The four values
of `case:Item Category` are `3-way match, invoice before GR` (221,010 cases), `3-way match, invoice
after GR` (15,182), `Consignment` (14,498) and `2-way match` (1,044); they sum to 251,734.

**The two `\authornote` confirmations the manuscript asks for are now discharged by the log itself.**
Every outcome activity Section 5.2 names is present under the name the manuscript uses: `Cancel
Invoice Receipt` (6,471 cases), `Cancel Goods Receipt` (2,470), `Set Payment Block` (122), `Remove
Payment Block` (55,839), `Vendor creates debit memo` (5,988), `Change Price` (11,224), `Change
Quantity` (17,590), `Delete Purchase Order Item` (8,839). The correction activities the reference
set of Section 4.2 excludes are present under the same names. The `\authornote` placeholders
themselves are left in place, the manuscript being frozen; this paragraph is the confirmation.

**Still not obtained as of the 2026-09-20 morning run**: BPI 2017, IEEE-CIS, Bank Account Fraud,
PaySim. Two of these four arrived later the same day; see the second addendum below.

## Obtained 2026-09-20 (second addendum): Bank Account Fraud and PaySim

Both files were fetched through a browser on the author's own machine (Kaggle is blocked at this
analysis environment's egress gateway) and installed under `data/raw/`. Both were re-hashed and re-counted **in
this run** before use; the hashes below are the ones this container computed, and they agree with
the values the author supplied.

| File (`data/raw/…`) | Bytes | SHA-256 (recomputed here) | Rows | Cols | Manuscript Table 2 | Verified |
|---|---|---|---|---|---|---|
| `baf/Base.csv` | 213,427,735 | `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809` | 1,000,000 | 32 | 1 M rows, ~1.1% positive | rows **exact**; `fraud_bool` sum = **11,029** = **1.1029%**, matching the "$\sim$1.1\%" of Table 2; schema is the published Base variant (`income`, `employment_status`, `velocity_6h/24h/4w`, `device_os`, `month`, …) |
| `paysim/PS_20174392719_1491204439457_log.csv` | 493,534,783 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` | 6,362,620 | 11 | 6,362,620 rows | rows **exact**; canonical schema `step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud`; `isFraud` sum = **8,213** (0.129%) |

Canonical sources: Bank Account Fraud — Jesus et al., *Turning the Tables: Biased, Imbalanced,
Dynamic Tabular Datasets for ML Evaluation*, NeurIPS 2022 Datasets and Benchmarks, distributed via
Kaggle `sgpjesus/bank-account-fraud-dataset-neurips-2022`; PaySim — Lopez-Rojas, Elmir & Axelsson
(2016), distributed via Kaggle `ealaxi/paysim1`.

**Bank Account Fraud: only the Base variant.** Variants I–V exist in the author's download but were
not extracted into `data/raw/baf/`. Base is the canonical variant and is the one used here; the
manuscript's Table 2 row says "6 variants", and the five bias-controlled variants are therefore
**not** exercised in this run. Recorded in `DEVIATIONS.md` 7.2.

**BAF month distribution** (the split variable), verified here: months 0–7 hold 132,440 / 127,620 /
136,979 / 150,936 / 127,691 / 119,323 / 108,168 / 96,843 records with positive rates rising from
1.13% to 1.47%. The temporal split used is months 0–5 for training and months 6–7 for testing —
the split the BAF paper itself uses and the `splits: {type: temporal}` rule of
`configs/default.yaml`. `month` is the split variable and is not a feature.

**BAF carries no monetary amount.** No field records a loss or a transaction value. The value proxy
adopted for the value-weighted metric is `proposed_credit_limit` (12 distinct values, 190–2,100;
mean 512 over all applications, 834 over fraudulent ones) — defined, justified and recorded in
`DEVIATIONS.md` 7.1 and in the note to Table 3, with count-based recall reported beside every
value-weighted cell in `RESULTS_REPORT.md`.

**PaySim is the host `configs/default.yaml` names** (`datasets: [… paysim …]`). The E7 injection
gate and the H4 value-at-budget endpoint were re-run on it (`results/e7_gate_paysim.json`,
`results/h4_primary_endpoint_paysim.json`); the ULB-hosted versions are kept and both are reported.

**Still not obtained**: BPI 2017 (4TU, blocked), Home Credit (Paper 1's dependency, not needed
here). Their slots stay open.

## Obtained 2026-09-20 (third addendum): IEEE-CIS Fraud Detection

Installed by the author at `data/raw/ieee_cis/` after acceptance of the competition rules; the
host (Kaggle) is blocked at the analysis environment's egress gateway, so the files were fetched on the
author's machine and copied in. Verified here on 2026-09-20 before use.

| File | Bytes | Rows x cols | SHA-256 |
|---|---|---|---|
| `train_transaction.csv` | 683,351,067 | **590,540 x 394** | `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642` |
| `train_identity.csv` | 26,529,680 | **144,233 x 41** | `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c` |

**Verification against the manuscript.** Table 2 states 590,540 transactions at a 3.5% fraud rate.
The file holds exactly 590,540 data rows; `isFraud == 1` in **20,663** of them, a rate of
**3.4990%**, which rounds to the 3.5% the table states. The 394 columns are the published schema
(`TransactionID`, `isFraud`, `TransactionDT`, `TransactionAmt`, `ProductCD`, `card1`–`card6`,
`addr1`–`addr2`, `dist1`–`dist2`, `P_emaildomain`, `R_emaildomain`, `C1`–`C14`, `D1`–`D15`,
`M1`–`M9`, `V1`–`V339`). The identity file holds 144,233 rows over 41 columns
(`TransactionID`, `id_01`–`id_38`, `DeviceType`, `DeviceInfo`) and joins to **24.4239%** of the
transactions. Source: IEEE-CIS Fraud Detection competition (Vesta Corporation / IEEE Computational
Intelligence Society, 2019), cited in the manuscript as `ieeecis2019`.

**IEEE-CIS carries a real monetary field.** `TransactionAmt` is an observed transaction amount
(median 68.769, mean 134.77, max 31,937.39; mean 149.24 over fraudulent transactions against
134.51 over legitimate ones). It is the only fraud column of Table 3 whose value weighting is an
observed amount: the Bank Account Fraud column uses `proposed_credit_limit` as an exposure proxy
(`DEVIATIONS.md` 7.1), the vehicle column uses the midpoints of a coded price band
(`DEVIATIONS.md` 7.2), and the Medicare column uses total reimbursement at provider level. This is
stated in the note to Table 3 and in `DEVIATIONS.md` 7.8.

**`TransactionDT` is a time delta, not a timestamp.** It runs from 86,400 to 15,811,131 seconds,
a span of 182.0 days, with no stated origin. It is used **purely as an ordering variable**: rows
are ordered by it and cut at its 75th percentile, 11,246,620 s (day 129.2 of 182.0), giving
442,905 training rows (3.514% fraudulent) and 147,635 test rows (3.454% fraudulent) — the same
75/25 temporal proportion as the BAF arm. `TransactionDT` is never a feature. Recorded in
`results/ieee_cis_design.json`.

**The 75.6% of transactions with no identity record are kept, not dropped.** The join is a left
join on `TransactionID`; the 446,307 transactions with no identity row keep their 40 identity
columns missing, and a binary `has_identity` feature records the fact. Missing entries are then
imputed by the training-window median of their column, so no transaction is discarded. Of the
432 features used, **no column is entirely missing within the training window** and none is
constant there — the many sparse `V` columns the file is known for are sparse, not empty, in this
window — and 115,523,073 cells (45.3% of the matrix) are imputed. The same imputed matrix is given
to all five detectors so that the comparison is on equal information: LightGBM would otherwise use
native `NaN` handling that the isolation forest, the autoencoder and the neural control cannot.

**Not exercised**: `test_transaction.csv` / `test_identity.csv` (unlabelled competition test set)
and `sample_submission.csv` were not installed and are not needed; every IEEE-CIS result here is
computed within `train_transaction.csv`'s own temporal split.

## Obtained 2026-09-20 (evening addendum): BPI Challenge 2017

| File (`data/raw/bpi2017/`) | Bytes | SHA-256 | Cases | Events | Manuscript Table 2 | Verified |
|---|---|---|---|---|---|---|
| `BPI_Challenge_2017.xes.gz` | 29,658,747 | of the gzip container `183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249`; of the **uncompressed** XES (578,941,403 bytes) `d653a36d36fac668638d65c90b803670bab6e599aa23e3f7dd4f0d5d0b216b1c` | 31,509 | 1,202,267 | 31,509 applications | **exact on both counts** |
| `bpi2017_events.csv` | 200,115,057 | `5a13474b8cfec85587b68923e07f6c9af41096389bf782ec8b6ebd0b205f5ca4` | 31,509 | 1,202,267 | — | derived from the XES by `code_run/xes_to_csv.py` (streaming `iterparse`) |

Source: data.4tu.nl, CC BY 4.0, retrieved 2026-09-20 through a browser on the author's own machine
(the host is blocked at the analysis environment's egress gateway, 403 on CONNECT).

**Independent verification in this run.** `code_run/e27_bpi2017.py` re-read the flat file and
counted **1,202,267 events over 31,509 applications**, matching the published figures and Table 2
exactly, and enumerated **26 distinct activity labels** (`results/bpi2017_verification.json`). The
uncompressed-XES SHA-256 above was recomputed in this session by streaming the gzip container, and
it matches the digest recorded for the 4TU distribution. Activity labels ignore
`lifecycle:transition`, as `e9`/`e11` do for BPI 2019; the log has 15,930 trace variants on that
reading and 5,623 on complete events only, and both are reported.

**Role.** The manuscript assigns this log the process-generality check and nothing else, and
nothing else was computed from it: no dissociation cell, no value-at-budget figure, no adversary
run. `results/bpi2017_generality_check.json`, `bpi2017_noise_grid.csv` and
`bpi2017_alignment_cost.json` are its only outputs.

## Consequence

All eight datasets the manuscript names have now been obtained: IEEE-CIS, ULB, vehicle claims,
Medicare, Bank Account Fraud and PaySim on the fraud/benchmark side, and the BPI Challenge 2019
and 2017 logs on the leakage/process side. The endpoints that remain "not computed" are those
needing human participants (H5), the feature-tokenised transformer on IEEE-CIS and on the leakage
side, and the parts of the conformance channel — the noise-threshold grid, the cost-weight
sensitivity and the two-way-match flow — that the compute budget or the flow's four positives put
out of reach. PyTorch is no longer a constraint: `torch` 2.14.0 is importable in this container and
the real transformer was fitted on three fraud datasets.
