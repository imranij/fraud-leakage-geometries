#!/usr/bin/env bash
# Full analysis run. Executes end to end from this directory.
# 2026-09-19  fraud side (e1-e8)
# 2026-09-20  leakage side (e9-e13) and the datasets that arrived that day (e14-e19)
# 2026-09-20  IEEE-CIS arrived last (e20-e21)
# 2026-09-20  evening: real FTT, optimal alignments, the H4 pairing, BPI 2017 (e22-e28)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p ../results ../results/cache ../logs
bash fetch_data.sh                                              # ULB, vehicle claims, Medicare (verified SHA-256)
# data/raw/bpi2019, data/raw/baf and data/raw/paysim are installed by the author (hosts blocked
# at the analysis environment's network gateway); DATA_PROVENANCE.md records their SHA-256 and counts.

# ---- fraud side --------------------------------------------------------------
python3 e2_h2_simulation.py       2>&1 | tee ../logs/e2.log      # H2 simulation (S3)
python3 e1_dissociation.py        2>&1 | tee ../logs/e1.log      # H1(c): vehicle, Medicare, ULB
python3 e14_baf_dissociation.py   2>&1 | tee ../logs/e14_baf.log # H1(c): BAF column (value proxy, 7.1)
python3 e20_ieee_dissociation.py  2>&1 | tee ../logs/e20_ieee.log   # H1(c): IEEE-CIS column (real TransactionAmt, 8.1-8.4); caches matrices
python3 e21_ieee_adversary.py     2>&1 | tee ../logs/e21_ieee_adv.log # H3(a,b) on IEEE-CIS, restricted mutable set (8.5)
python3 e4_durability.py --retrain-eval legacy 2>&1 | tee ../logs/e4.log      # H3(a,b) adaptive adversary; frozen arm carries the endpoint (8.7)
python3 e4_durability.py --retrain-eval fixed --datasets medicare --cells cross 2>&1 | tee ../logs/e4_fixed.log  # retrained arm re-evaluated out of order (RETRAINED_ARM_NOTE.md)
python3 e3_value_at_budget.py     2>&1 | tee ../logs/e3.log      # H4, ULB host (uncorrected periods)
python3 e18_h4_period_fix.py      2>&1 | tee ../logs/e18_fix.log # H4, ULB host, period defect fixed (7.5)
python3 e16_paysim_host.py        2>&1 | tee ../logs/e16_paysim.log # E7 gate on the PaySim host (7.4)
python3 e19_paysim_h4_fix.py      2>&1 | tee ../logs/e19_paysim_fix.log # H4 on the PaySim host, periods fixed (7.4, 7.5)
python3 e7_injection_gate.py      2>&1 | tee ../logs/e7.log      # injected side of the E7 gate (ULB host)
python3 e8_latency.py             2>&1 | tee ../logs/e8.log      # throughput per channel

# ---- leakage side (BPI Challenge 2019) ---------------------------------------
python3 e9_leakage_prep.py        2>&1 | tee ../logs/e9_prep.log     # one chunked pass -> results/cache/
python3 e10_leakage_population.py 2>&1 | tee ../logs/e10_pop.log     # Table S5, windows A/B/C, typology
python3 e13_gate_and_drift.py     2>&1 | tee ../logs/e13_gate.log    # E7 gate verdict, H3(c), drift curve
python3 e11_conformance.py        2>&1 | tee ../logs/e11_conf.log    # optimal alignments -- DOES NOT COMPLETE (4.1)
python3 e12_leakage_detectors.py  2>&1 | tee ../logs/e12_det.log     # leakage columns of Table 3
python3 e15_h1a_diagnosis.py      2>&1 | tee ../logs/e15_diag.log    # H1(a) diagnosis: count recall, value concentration, value-only null
python3 e17_conformance_replay.py 2>&1 | tee ../logs/e17_replay.log  # CONF via token replay + H1(b) (substitution 7.6)

# ---- the run of 2026-09-20 (evening): the transformer, the alignments, the pairing ----
# e22 fits the REAL feature-tokenised transformer (torch 2.14.0 is importable in this container,
# so substitution 2.1 is retired for the arms it runs).  The vehicle arm defaults to 40 epochs:
# its first run at 8 epochs was undertrained and the training loss was still falling.
VEHICLE_EPOCHS=${VEHICLE_EPOCHS:-40} python3 e22_ftt.py vehicle medicare baf 2>&1 | tee ../logs/e22_ftt.log
python3 e23_alignments.py         2>&1 | tee ../logs/e23_align.log   # CONF/GBDT+phi/H1(b) by OPTIMAL ALIGNMENT on a declared variant sample (ADDENDUM A.2)
python3 e24_h4_pairing.py         2>&1 | tee ../logs/e24_h4pair.log  # H4 on the pre-specified pairing BPI 2019 x IEEE-CIS (ADDENDUM B.1)
python3 e25_kappa_cells.py        2>&1 | tee ../logs/e25_kappa.log   # the kappa cells of the H3 grid, frozen arm, three seeds (ADDENDUM B.2)
python3 e26_flow2_column.py       2>&1 | tee ../logs/e26_flow2.log   # "3-way match, invoice before GR" column on the FULL population (ADDENDUM B.3)
python3 e27_bpi2017.py            2>&1 | tee ../logs/e27_bpi2017.log # BPI 2017 process-generality check and alignment cost (ADDENDUM B.4)
python3 e28_e7_ieee.py            2>&1 | tee ../logs/e28_e7_ieee.log # E7 gate on the IEEE-CIS host, the third named host (ADDENDUM B.5)
# NOT run and therefore not in this script: the FTT on IEEE-CIS (432 feature tokens) and on the
# leakage side; their cells stay slots and no MLP number is relabelled.

# ---- figures ------------------------------------------------------------------
python3 make_figures.py           2>&1 | tee ../logs/figs.log
