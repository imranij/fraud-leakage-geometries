# Explainable artificial intelligence and process conformance for fraud and payment leakage detection

Imran and Pir Noman Ahmad, King Fahd University of Petroleum and Minerals, Dhahran, Saudi Arabia.

This repository holds the analysis code and every computed result behind the paper of the same title. The paper argues that fraud and payment leakage are different objects: fraud is a record an adversary places inside the distribution of legitimate records, and leakage is a trace that departs from the normative business process. The scripts here test that claim on a public purchase-to-pay event log (BPI Challenge 2019), a loan-application log (BPI Challenge 2017) and six public fraud datasets. They run the dissociation test between conformance checking and supervised learning, the adaptive-adversary durability experiment, the stationarity check on the leakage side, and the budgeted two-queue triage.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | One script per analysis (`e1_` to `e28_`), the shared loaders and metrics in `common.py`, the data-figure script `make_figures.py`, and `run_all.sh`, which runs everything in order |
| `lib/` | The LeakBench injection protocol (`src/leakbench/`), the value-at-budget metrics and triage policies (`src/triage/`), their configuration and tests |
| `results/` | Every number the paper reports, one CSV or JSON file per analysis. The largest file, `conf_align_delta.csv`, is stored compressed as `conf_align_delta.csv.gz`; decompress it before re-reading |
| `figures/` | The three data figures (adversarial decay, process drift, the cross-channel simulation) and the palette they share |
| `docs/` | Data provenance with SHA-256 hashes, and the deviations log, which records every departure from the pre-specified design |
| `data/` | Empty. The datasets are public and are not redistributed here (see below) |

## Setup

Python 3.11 was used. Install the requirements with `pip install -r requirements.txt`.

Download each dataset from its public source and place it at the path below. `docs/DATA_PROVENANCE.md` gives the source, the byte count and the SHA-256 hash of every file, so a download can be checked before use. `code/fetch_data.sh` downloads and verifies the ULB, vehicle-claims and Medicare files automatically.

| Dataset | Path |
|---|---|
| BPI Challenge 2019 (4TU.ResearchData) | `data/raw/bpi2019/BPI_Challenge_2019.xes.gz` |
| BPI Challenge 2017 (4TU.ResearchData) | `data/raw/bpi2017/BPI_Challenge_2017.xes.gz` |
| IEEE-CIS Fraud Detection | `data/raw/ieee_cis/train_transaction.csv`, `train_identity.csv` |
| Bank Account Fraud, base variant | `data/raw/baf/Base.csv` |
| PaySim | `data/raw/paysim/PS_20174392719_1491204439457_log.csv` |
| ULB credit card | `data/raw/ulb/creditcard.csv` |
| Vehicle insurance claims | `data/raw/carclaims/fraud_oracle.csv` |
| Medicare provider fraud | `data/raw/medicare/Train*-1542865627584.csv` (four files) |

`code/xes_to_csv.py` converts each XES log into the event-level CSV the leakage scripts read.

## Reproduction

```
cd code
bash run_all.sh
```

`run_all.sh` lists the scripts in run order, fraud side first, then the leakage side, then the data figures. Each script writes its outputs to `results/`. The analyses were run on 19 and 20 September 2026 on a two-core Linux machine with Python 3.11. Optimal alignment on BPI Challenge 2019 is the slowest step, at 10 to 15 seconds per trace variant in the tail of the log, which is why it runs on a declared, seeded sample of variants (`docs/DEVIATIONS.md`).

## What was verified

On 26 November 2026, from a fresh copy of this repository:

- every Python file in `code/` and `lib/` passed a syntax check (66 files);
- the four LeakBench tests in `lib/tests/` passed;
- `make_figures.py`, run on the committed results, reproduced the three data figures at the same page size and with identical text.

The full analysis pipeline was not re-run for this release, because it needs the 2.2 GB of raw data and several hours of compute.

## Not included

- The raw datasets, which are public and are redistributed by their own hosts.
- Intermediate caches that the scripts rebuild from the raw data.
- The prompt templates of the narrative-rewriting adversary variant, which are withheld under the release policy described in the paper and are available to researchers on request for evaluating detection systems.

## Licence

No licence is granted. The code and results may be read, but not reused or redistributed without the authors' permission.

## How to cite

Imran and Pir Noman Ahmad. Explainable artificial intelligence and process conformance for fraud and payment leakage detection. Manuscript under review, 2026. The Zenodo DOI of this archive will be added here once it is minted.
