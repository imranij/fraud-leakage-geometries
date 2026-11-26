#!/usr/bin/env bash
# Paper 2 -- fetch the three public fraud files the run used (2026-09-19), from the GitHub mirrors
# recorded in DATA_PROVENANCE.md, and verify their SHA-256. BPI 2019 (4TU), IEEE-CIS, BAF and PaySim
# were not obtainable; place them under ../data/raw/ by hand for a follow-up run.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/ulb data/raw/carclaims data/raw/medicare
get() { [ -s "$2" ] || curl -sSL -o "$2" "$1"; }
get https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv data/raw/ulb/creditcard.csv
get https://raw.githubusercontent.com/Shritik12345/-vehicle-insurance-fraud-detection/main/fraud_oracle.csv data/raw/carclaims/fraud_oracle.csv
for f in Train Train_Beneficiarydata Train_Inpatientdata Train_Outpatientdata; do
  get "https://raw.githubusercontent.com/sagnikgh1899/FraudDetection/main/FraudDetection/data/${f}-1542865627584.csv" "data/raw/medicare/${f}-1542865627584.csv"
done
sha256sum -c - <<'SUMS'
33a178be65174943a07d1736acf5898f6c8e1fa80950dbbe598c17eda968ec59  data/raw/ulb/creditcard.csv
8b6aa59764ef4f8b058598d3e8f325ef3623f52f4b1cd4ef946baebb5ccfa9a6  data/raw/carclaims/fraud_oracle.csv
09315a0e0c262ddfb91df15df76b196a80de2ff1001fad46b362b562df2e7e43  data/raw/medicare/Train-1542865627584.csv
1fb7fb14efeaae04b970bfeb3176cdf938c755748217377109081d6dae01ecfd  data/raw/medicare/Train_Beneficiarydata-1542865627584.csv
395bc2d01a102bf44a26ba4b36c63e7086943eaf2d5d1489c776bf47d1613ac0  data/raw/medicare/Train_Inpatientdata-1542865627584.csv
f532ed7e25f50d6abf3ba89ceeb2a23e2a352792f90d086b13999a561fd23889  data/raw/medicare/Train_Outpatientdata-1542865627584.csv
SUMS
