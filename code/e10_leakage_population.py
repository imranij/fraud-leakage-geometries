"""e10_leakage_population.py -- the outcome-labelled leakage population of BPI 2019,
the detector's deviation typology, and the overlap between them.

Label definition (main_journal.tex Sec. 5.2, verbatim): a case is leakage-positive if the log
records, AFTER an invoice was cleared or a payment released, an activity consistent with the
payment having gone wrong -- cancellation of an invoice receipt, cancellation of a goods receipt,
setting or removal of a payment block, a vendor debit memo, a price or quantity change after
clearing, or deletion of the item after clearing.

The three window operationalisations of "after clearing", FIXED BEFORE COMPUTING (DEVIATIONS 5):
  Window A (narrowest) : outcome activity strictly after the LAST `Clear Invoice` in the case
  Window B (PRIMARY)   : outcome activity strictly after the FIRST `Clear Invoice` in the case
  Window C (widest)    : outcome activity anywhere in the case (clearing not required)

Deviation typology (Sec. 4.2, six types) -- the DETECTOR's rules, never used as labels.
"""
import os, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
CACHE = os.path.join(PAPER, "results", "cache"); RES = os.path.join(PAPER, "results")

ev = np.load(os.path.join(CACHE, "events.npz"))
ci, ac, ts, vl = ev["case_idx"], ev["act_code"], ev["ts"], ev["val"]
cases = pd.read_pickle(os.path.join(CACHE, "cases.pkl"))
acts = pd.read_csv(os.path.join(CACHE, "activities.csv"))
A = dict(zip(acts.activity, acts.code))
N = len(cases)

OUTCOME = {
    "Cancel Invoice Receipt":        "cancellation of an invoice receipt",
    "Cancel Goods Receipt":          "cancellation of a goods receipt",
    "Set Payment Block":             "setting of a payment block",
    "Remove Payment Block":          "removal of a payment block",
    "Vendor creates debit memo":     "vendor debit memo",
    "Change Price":                  "price change after clearing",
    "Change Quantity":               "quantity change after clearing",
    "Delete Purchase Order Item":    "deletion of the item after clearing",
}
CLEAR = A["Clear Invoice"]

# ---- per-case clearing times -------------------------------------------------
BIG = np.iinfo(np.int64).max
first_clear = np.full(N, BIG, dtype=np.int64)
last_clear = np.full(N, -1, dtype=np.int64)
m = ac == CLEAR
np.minimum.at(first_clear, ci[m], ts[m])
np.maximum.at(last_clear, ci[m], ts[m])
has_clear = last_clear >= 0

# ---- outcome flags per activity per window ----------------------------------
rows = []
posA = np.zeros(N, bool); posB = np.zeros(N, bool); posC = np.zeros(N, bool)
per_act = {}
for name in OUTCOME:
    code = A[name]
    sel = ac == code
    c_sel, t_sel = ci[sel], ts[sel]
    fa = np.zeros(N, bool); fb = np.zeros(N, bool); fc = np.zeros(N, bool)
    fc[c_sel] = True                                        # window C: anywhere
    kb = t_sel > first_clear[c_sel]; fb[c_sel[kb]] = True    # window B: after FIRST clear
    ka = (t_sel > last_clear[c_sel]) & has_clear[c_sel]      # window A: after LAST clear
    fa[c_sel[ka]] = True
    per_act[name] = (fa, fb, fc)
    posA |= fa; posB |= fb; posC |= fc

ft = cases["flow_type"].to_numpy()
FLOWS = ["3-way match, invoice after GR", "3-way match, invoice before GR", "2-way match", "Consignment"]
for name in OUTCOME:
    fa, fb, fc = per_act[name]
    for flow in FLOWS:
        k = ft == flow
        rows.append({"outcome_activity": name, "flow_type": flow, "n_cases_in_flow": int(k.sum()),
                     "window_A": int((fa & k).sum()), "window_B": int((fb & k).sum()),
                     "window_C": int((fc & k).sum())})
    rows.append({"outcome_activity": name, "flow_type": "ALL", "n_cases_in_flow": N,
                 "window_A": int(fa.sum()), "window_B": int(fb.sum()), "window_C": int(fc.sum())})
for flow in FLOWS + ["ALL"]:
    k = np.ones(N, bool) if flow == "ALL" else (ft == flow)
    rows.append({"outcome_activity": "ANY (union)", "flow_type": flow, "n_cases_in_flow": int(k.sum()),
                 "window_A": int((posA & k).sum()), "window_B": int((posB & k).sum()),
                 "window_C": int((posC & k).sum())})
tabS5 = pd.DataFrame(rows)
tabS5.to_csv(os.path.join(RES, "tabS5_leakage_population.csv"), index=False)

# ---- deviation typology (six types, Sec. 4.2) -------------------------------
def count_of(code):
    out = np.zeros(N, dtype=np.int32)
    np.add.at(out, ci[ac == code], 1)
    return out

n_ir   = count_of(A["Record Invoice Receipt"])
n_gr   = count_of(A["Record Goods Receipt"])
n_clr  = count_of(A["Clear Invoice"])
n_vinv = count_of(A["Vendor creates invoice"])

first_of = {}
for nm in ["Record Goods Receipt", "Record Invoice Receipt", "Create Purchase Order Item",
           "Vendor creates invoice"]:
    f = np.full(N, BIG, dtype=np.int64); s = ac == A[nm]
    np.minimum.at(f, ci[s], ts[s]); first_of[nm] = f

# value carried at GR events vs at IR events (Cumulative net worth EUR)
def maxabs_at(code):
    out = np.zeros(N); s = ac == code
    np.maximum.at(out, ci[s], np.abs(np.nan_to_num(vl[s])))
    return out
v_gr = maxabs_at(A["Record Goods Receipt"]); v_ir = maxabs_at(A["Record Invoice Receipt"])

T = {}
T["skipped goods receipt"]   = (n_clr > 0) & (n_gr == 0)
T["invoice before purchase order"] = (
    np.minimum(first_of["Record Invoice Receipt"], first_of["Vendor creates invoice"])
    < first_of["Create Purchase Order Item"])
T["duplicate invoice"]       = n_ir >= 2
T["repeated clearing"]       = n_clr >= 2
T["value mismatch"]          = (v_gr > 0) & (v_ir > 0) & (v_ir > v_gr * 1.005)

# atypical role: an event executed by a resource that performs that activity rarely (<0.1%)
res = pd.read_csv(os.path.join(PAPER, "data", "raw", "bpi2019", "bpi2019_events.csv"),
                  usecols=["org:resource", "concept:name"], dtype=str)
pair = res["concept:name"].astype(str) + "\x1f" + res["org:resource"].astype(str)
pc = pair.value_counts()
actc = res["concept:name"].value_counts()
share = pc / pc.index.to_series().str.split("\x1f").str[0].map(actc)
rare = set(share.index[share < 0.001])
is_rare = pair.isin(rare).to_numpy()
T["atypical role"] = np.zeros(N, bool)
np.logical_or.at(T["atypical role"], ci, is_rare)
del res, pair

typed_any = np.zeros(N, bool)
for v in T.values():
    typed_any |= v

typ_rows = []
for tname, v in T.items():
    for flow in FLOWS + ["ALL"]:
        k = np.ones(N, bool) if flow == "ALL" else (ft == flow)
        typ_rows.append({"deviation_type": tname, "flow_type": flow,
                         "n_cases": int((v & k).sum()), "n_in_flow": int(k.sum()),
                         "share": float((v & k).sum() / max(k.sum(), 1))})
for flow in FLOWS + ["ALL"]:
    k = np.ones(N, bool) if flow == "ALL" else (ft == flow)
    typ_rows.append({"deviation_type": "ANY (union)", "flow_type": flow,
                     "n_cases": int((typed_any & k).sum()), "n_in_flow": int(k.sum()),
                     "share": float((typed_any & k).sum() / max(k.sum(), 1))})
pd.DataFrame(typ_rows).to_csv(os.path.join(RES, "leakage_deviation_typology.csv"), index=False)

# ---- overlap between labels (window B) and typology --------------------------
CORRECTION = ["Cancel Invoice Receipt", "Cancel Goods Receipt", "Vendor creates debit memo",
              "Cancel Subsequent Invoice", "Delete Purchase Order Item"]
corr = np.zeros(N, bool)
for nm in CORRECTION:
    corr[ci[ac == A[nm]]] = True

ov = {
    "n_cases": int(N),
    "positives_window_A": int(posA.sum()), "positives_window_B": int(posB.sum()),
    "positives_window_C": int(posC.sum()),
    "rate_window_B": float(posB.mean()),
    "added_by_widest": int(posC.sum() - posB.sum()),
    "removed_by_narrowest": int(posB.sum() - posA.sum()),
    "spread_across_windows_abs": int(posC.sum() - posA.sum()),
    "spread_across_windows_share_of_primary": float((posC.sum() - posA.sum()) / max(posB.sum(), 1)),
    "typed_any": int(typed_any.sum()),
    "share_of_positives_carrying_a_typed_deviation": float((posB & typed_any).sum() / max(posB.sum(), 1)),
    "share_of_typed_ending_in_a_correction": float((typed_any & corr).sum() / max(typed_any.sum(), 1)),
    "share_of_typed_that_are_positive": float((typed_any & posB).sum() / max(typed_any.sum(), 1)),
    "jaccard_label_typology": float((posB & typed_any).sum() / max((posB | typed_any).sum(), 1)),
}
for flow in FLOWS:
    k = ft == flow
    ov[f"rate_window_B::{flow}"] = float((posB & k).sum() / max(k.sum(), 1))
    ov[f"n_window_B::{flow}"] = int((posB & k).sum())
json.dump(ov, open(os.path.join(RES, "leakage_population_summary.json"), "w"), indent=1)

lab = pd.DataFrame({"case_idx": np.arange(N), "flow_type": ft,
                    "pos_A": posA, "pos_B": posB, "pos_C": posC,
                    "typed_any": typed_any, "correction": corr,
                    "value": cases["value_absmax"].to_numpy(),
                    "t_first": cases["t_first"].to_numpy(), "t_last": cases["t_last"].to_numpy()})
for tname, v in T.items():
    lab["T_" + tname.replace(" ", "_")] = v
lab.to_pickle(os.path.join(CACHE, "labels.pkl"))

print(json.dumps(ov, indent=1))
print(tabS5[tabS5.flow_type == "ALL"].to_string(index=False))
