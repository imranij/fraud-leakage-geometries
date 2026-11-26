"""e9_leakage_prep.py -- one streaming pass over bpi2019_events.csv.

Builds a compact event-array + case-attribute store under results/cache/ that every
downstream leakage script reads.  Memory-safe: chunked read, explicit dtypes,
never more than one chunk of the 422 MB CSV resident at a time.

Outputs
  results/cache/events.npz          case_idx int32, act_code int16, ts int64(ns), row order = file order
  results/cache/cases.pkl       one row per case: id, flow type, value, vendor, n_events, t_first, t_last
  results/cache/activities.csv      activity label -> code, global frequency
  results/leakage_activity_census.csv  same, for the record
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
CSV = os.path.join(PAPER, "data", "raw", "bpi2019", "bpi2019_events.csv")
CACHE = os.path.join(PAPER, "results", "cache")
RES = os.path.join(PAPER, "results")
os.makedirs(CACHE, exist_ok=True)

USECOLS = [
    "case:concept:name", "concept:name", "time:timestamp",
    "case:Item Category", "Cumulative net worth (EUR)",
    "case:Vendor", "case:Spend area text", "case:Document Type",
    "case:GR-Based Inv. Verif.", "case:Goods Receipt", "case:Item Type",
    "org:resource", "case:Company",
]
DTYPES = {
    "case:concept:name": "str", "concept:name": "category", "case:Item Category": "category",
    "Cumulative net worth (EUR)": "float32", "case:Vendor": "str", "case:Spend area text": "category",
    "case:Document Type": "category", "case:GR-Based Inv. Verif.": "str", "case:Goods Receipt": "str",
    "case:Item Type": "category", "org:resource": "str", "case:Company": "category",
}

t0 = time.time()
case_ids = {}          # case id -> index
case_rows = []         # per-case attributes in index order
act_ids = {}           # activity -> code
ev_case, ev_act, ev_ts, ev_val = [], [], [], []

CH = 400_000
n_rows = 0
for ch in pd.read_csv(CSV, usecols=USECOLS, dtype=DTYPES, chunksize=CH):
    n_rows += len(ch)
    ts = pd.to_datetime(ch["time:timestamp"], format="ISO8601", utc=True).astype("int64").to_numpy()
    acts = ch["concept:name"].astype(str).to_numpy()
    cids = ch["case:concept:name"].to_numpy()
    vals = ch["Cumulative net worth (EUR)"].to_numpy()
    ci = np.empty(len(ch), dtype=np.int32)
    ai = np.empty(len(ch), dtype=np.int16)
    fc = ch["case:Item Category"].astype(str).to_numpy()
    vend = ch["case:Vendor"].astype(str).to_numpy()
    spend = ch["case:Spend area text"].astype(str).to_numpy()
    dtp = ch["case:Document Type"].astype(str).to_numpy()
    grbiv = ch["case:GR-Based Inv. Verif."].astype(str).to_numpy()
    grf = ch["case:Goods Receipt"].astype(str).to_numpy()
    itp = ch["case:Item Type"].astype(str).to_numpy()
    comp = ch["case:Company"].astype(str).to_numpy()
    for k in range(len(ch)):
        c = cids[k]
        j = case_ids.get(c)
        if j is None:
            j = len(case_rows)
            case_ids[c] = j
            case_rows.append((c, fc[k], vend[k], spend[k], dtp[k], grbiv[k], grf[k], itp[k], comp[k]))
        ci[k] = j
        a = acts[k]
        b = act_ids.get(a)
        if b is None:
            b = len(act_ids); act_ids[a] = b
        ai[k] = b
    ev_case.append(ci); ev_act.append(ai); ev_ts.append(ts); ev_val.append(vals.astype(np.float32))
    print(f"  {n_rows:,} rows, {len(case_rows):,} cases, {len(act_ids)} activities, {time.time()-t0:.0f}s", flush=True)

ev_case = np.concatenate(ev_case); ev_act = np.concatenate(ev_act)
ev_ts = np.concatenate(ev_ts); ev_val = np.concatenate(ev_val)
print(f"TOTAL events={len(ev_case):,} cases={len(case_rows):,} activities={len(act_ids)}", flush=True)

np.savez_compressed(os.path.join(CACHE, "events.npz"),
                    case_idx=ev_case, act_code=ev_act, ts=ev_ts, val=ev_val)

cases = pd.DataFrame(case_rows, columns=["case_id", "flow_type", "vendor", "spend_area",
                                         "doc_type", "gr_based_inv_verif", "goods_receipt_flag",
                                         "item_type", "company"])
# per-case aggregates
order = np.argsort(ev_case, kind="stable")
cs = ev_case[order]
bounds = np.searchsorted(cs, np.arange(len(case_rows) + 1))
cases["n_events"] = np.diff(bounds)
tss = ev_ts[order]; vls = ev_val[order]
t_first = np.empty(len(case_rows), dtype=np.int64); t_last = np.empty(len(case_rows), dtype=np.int64)
v_max = np.empty(len(case_rows), dtype=np.float64); v_absmax = np.empty(len(case_rows), dtype=np.float64)
for j in range(len(case_rows)):
    a, b = bounds[j], bounds[j + 1]
    seg_t = tss[a:b]; seg_v = vls[a:b]
    t_first[j] = seg_t.min(); t_last[j] = seg_t.max()
    v_max[j] = float(np.nanmax(seg_v)) if b > a else np.nan
    v_absmax[j] = float(np.nanmax(np.abs(seg_v))) if b > a else np.nan
cases["t_first"] = t_first; cases["t_last"] = t_last
cases["value_max"] = v_max; cases["value_absmax"] = v_absmax
cases.to_pickle(os.path.join(CACHE, "cases.pkl"))

acts_df = pd.DataFrame({"activity": list(act_ids.keys()), "code": list(act_ids.values())})
freq = np.bincount(ev_act, minlength=len(act_ids))
acts_df["n_events"] = freq[acts_df["code"].to_numpy()]
ucase = pd.DataFrame({"code": ev_act, "case": ev_case}).drop_duplicates()
ncases = ucase.groupby("code").size()
acts_df["n_cases"] = acts_df["code"].map(ncases).fillna(0).astype(int)
acts_df = acts_df.sort_values("n_events", ascending=False)
acts_df.to_csv(os.path.join(CACHE, "activities.csv"), index=False)
acts_df.to_csv(os.path.join(RES, "leakage_activity_census.csv"), index=False)

meta = {"n_events": int(len(ev_case)), "n_cases": int(len(case_rows)),
        "n_activities": int(len(act_ids)),
        "flow_types": cases["flow_type"].value_counts().to_dict(),
        "csv_rows_read": int(n_rows), "seconds": round(time.time() - t0, 1)}
json.dump(meta, open(os.path.join(RES, "leakage_prep_meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
