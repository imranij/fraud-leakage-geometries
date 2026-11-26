"""Figures 4 and 5, in the serif style and palette of the three designed figures."""
from __future__ import annotations

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
sys.path.insert(0, FIG)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import palette as pal
import common as C

pal.set_rcparams(base_size=8.3)


def fig4():
    raw = pd.read_csv(os.path.join(C.RES, "fig4_h2_transfer_raw.csv"))
    g = raw.groupby("pi")["gain"]
    mean, sd, pis = g.mean(), g.std(ddof=1), sorted(raw.pi.unique())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))

    ax = axes[0]
    m = np.array([mean[p] for p in pis]); s = np.array([sd[p] for p in pis])
    ax.axhline(0, color=pal.SLATE, lw=0.7, ls=(0, (3, 2)))
    ax.fill_between(pis, m - s, m + s, color=pal.FRAUD_FILL, alpha=0.9, lw=0)
    ax.plot(pis, m, "-o", color=pal.FRAUD, ms=3.6, lw=1.4)
    for p in pis:
        ax.plot([p] * len(raw[raw.pi == p]), raw.loc[raw.pi == p, "gain"], ".",
                color=pal.SLATE, ms=2.6, alpha=0.7)
    ax.set_xlabel(r"weak-control entry probability $\pi$")
    ax.set_ylabel(r"gain in recall at 1% FPR from $\phi(\sigma)$")
    ax.set_title("(a) Transfer gain, bands over five seeds", loc="left", color=pal.INK)
    ax.set_xticks(pis)

    ax = axes[1]
    w = 0.34
    x = np.arange(len(pis))
    r0 = [raw.loc[raw.pi == p, "recall_records_only"].mean() for p in pis]
    r1 = [raw.loc[raw.pi == p, "recall_with_phi"].mean() for p in pis]
    e0 = [raw.loc[raw.pi == p, "recall_records_only"].std(ddof=1) for p in pis]
    e1 = [raw.loc[raw.pi == p, "recall_with_phi"].std(ddof=1) for p in pis]
    ax.bar(x - w / 2, r0, w, yerr=e0, color=pal.BOX_FILL, edgecolor=pal.INK, lw=0.7,
           error_kw=dict(lw=0.7, capsize=2), label=r"records only")
    ax.bar(x + w / 2, r1, w, yerr=e1, color=pal.LEAK_FILL, edgecolor=pal.LEAK, lw=0.7,
           error_kw=dict(lw=0.7, capsize=2), label=r"records $+\ \phi(\sigma)$")
    ax.set_xticks(x); ax.set_xticklabels([str(p) for p in pis])
    ax.set_xlabel(r"$\pi$"); ax.set_ylabel(r"recall at 1% FPR")
    ax.set_title("(b) Levels behind the gain", loc="left", color=pal.INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=2.5, width=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_h2_transfer.pdf"))
    plt.close(fig)
    print("wrote fig4")


def fig5():
    """Durability on the fraud side, FROZEN detector only, Medicare and IEEE-CIS.

    Revised 2026-09-21. The first version drew the retrained arm as a dashed line. That arm's
    evaluation was in-sample (e4_durability.py appended the adapted records to the training set
    and then scored the same records; DEVIATIONS.md 8.7), the manuscript withdrew it, and a figure
    must not show what the text withholds. IEEE-CIS is added because the Discussion compares the
    two datasets' decay (2.76 against 1.26 in value stolen) and a reader should see both.
    """
    frames = []
    for fn in ("fig5_durability_raw.csv", "fig5_durability_raw_IEEE.csv"):
        fp = os.path.join(C.RES, fn)
        if os.path.exists(fp):
            frames.append(pd.read_csv(fp))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[~raw.retrain.astype(bool)]                      # frozen arm only
    prim_q, prim_k = 0.05, 0.1
    style = {"Medicare": (pal.INK, "s"), "IEEE-CIS": (pal.FRAUD, "o")}
    dsets = [d for d in ("Medicare", "IEEE-CIS") if (raw.dataset == d).any()]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7))

    ax = axes[0]
    for ds in dsets:
        s = raw[(raw.dataset == ds) & (raw.q == prim_q) & (raw.kappa == prim_k)]
        g = s.groupby("round")["vw_recall"]
        m, sd = g.mean(), g.std(ddof=1).fillna(0)
        ax.fill_between(m.index, m - sd, m + sd, color=style[ds][0], alpha=0.12, lw=0)
        ax.plot(m.index, m.values, "-", color=style[ds][0], lw=1.3, label=ds)
    ax.set_xlabel("attack round $t$"); ax.set_ylabel(r"$R_t$: value-weighted recall at 1% FPR")
    ax.set_title("(a) Recall under adaptation", loc="left", color=pal.INK)
    ax.legend(frameon=False, fontsize=6.2, loc="best")

    ax = axes[1]
    for ds in dsets:
        s = raw[(raw.dataset == ds) & (raw.q == prim_q) & (raw.kappa == prim_k)]
        g = s.groupby("round")["value_stolen"].mean()
        v0 = g.iloc[0] if g.iloc[0] else np.nan
        ax.plot(g.index * 90 / 365.0, g.values / v0, "-", color=style[ds][0], lw=1.3, label=ds)
    ax.axhline(1.0, color=pal.SLATE, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlabel(r"calendar years ($\Delta = 90$ days per round)")
    ax.set_ylabel(r"$V_t / V_0$")
    ax.set_title("(b) Value stolen, calendar axis", loc="left", color=pal.INK)

    ax = axes[2]
    last = raw[raw["round"] == raw["round"].max()]
    for ds in dsets:
        col, mk = style[ds]
        s = last[last.dataset == ds]
        g = s[s.kappa == prim_k].groupby("q")["vw_recall"].mean()
        ax.plot(g.index, g.values, mk + "-", color=col, ms=3.4, lw=1.2, label=f"{ds}, by $q$")
        g2 = s[s.q == prim_q].groupby("kappa")["vw_recall"].mean()
        ax.plot(g2.index, g2.values, mk + "--", color=col, ms=3.4, lw=1.2, alpha=0.55,
                label=rf"{ds}, by $\kappa$")
    ax.set_xlabel(r"$q$ (solid) or $\kappa$ (dashed)")
    ax.set_ylabel(r"$R_T$ at the last round")
    ax.set_title("(c) Spread across the grid", loc="left", color=pal.INK)
    ax.legend(frameon=False, fontsize=6.0, loc="best")

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=2.5, width=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_durability.pdf"))
    plt.close(fig)
    print("wrote fig5 (frozen arm only; datasets: " + ", ".join(dsets) + ")")


def fig6():
    """Figure 6, added 2026-09-21: the leakage side of durability, which Figure 5 does not draw.

    Everything here is read from two files e13_gate_and_drift.py wrote: h3c_stationarity.json
    (the deviation-type mix in each half of the log and the permutation reference) and
    h3c_process_drift.csv (the monthly Jensen-Shannon divergence of that mix from the first
    third of the log, the typed share and the outcome-label positive rate, months with fewer
    than 200 cases already dropped). Nothing is computed here beyond what those files hold.
    """
    stat = json.load(open(os.path.join(C.RES, "h3c_stationarity.json")))
    drift = pd.read_csv(os.path.join(C.RES, "h3c_process_drift.csv"))
    names = {"T_skipped_goods_receipt": "skipped\nGR",
             "T_invoice_before_purchase_order": "invoice\nbefore PO",
             "T_duplicate_invoice": "duplicate\ninvoice",
             "T_repeated_clearing": "repeated\nclearing",
             "T_value_mismatch": "value\nmismatch",
             "T_atypical_role": "atypical\nrole"}
    types = list(names)
    m1 = np.array([stat["mix_first_half"][t] for t in types])
    m2 = np.array([stat["mix_second_half"][t] for t in types])

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # (a) the two halves, side by side
    ax = axes[0]
    x = np.arange(len(types)); w = 0.38
    ax.bar(x - w / 2, 100 * m1, w, color=pal.INK, label="first half", zorder=3)
    ax.bar(x + w / 2, 100 * m2, w, color=pal.LEAK, hatch="////", edgecolor="white",
           linewidth=0.4, label="second half", zorder=3)
    for xi, (a, b) in enumerate(zip(m1, m2)):
        if abs(b - a) >= 0.05:        # label only the moves worth reading
            ax.annotate(f"{100 * a:.0f}→{100 * b:.0f}", (xi, 100 * max(a, b) + 1.2),
                        ha="center", fontsize=6.2, color=pal.INK)
    ax.set_xticks(x); ax.set_xticklabels([names[t] for t in types], fontsize=6.0)
    ax.set_ylim(0, 100 * max(m1.max(), m2.max()) * 1.22)
    ax.set_ylabel("share of typed deviations (%)")
    ax.set_title("(a) Deviation-type mix by half of the log", loc="left", color=pal.INK)
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax.text(0.02, 0.72,
            f"JSD between halves {stat['jsd_between_halves']:.3f}\n"
            f"permutation 95th pct {stat['permutation_q95']:.5f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=pal.SLATE)

    # (b) monthly divergence from the first third of the log
    ax = axes[1]
    mon = pd.PeriodIndex(drift["month"], freq="M").to_timestamp()
    thin = drift["n_cases"] < 1000
    ax.plot(mon, drift["jsd_vs_first_third"], "-", color=pal.LEAK, lw=1.3, zorder=2)
    ax.plot(mon[~thin], drift.loc[~thin, "jsd_vs_first_third"], "o", color=pal.LEAK, ms=3.4, zorder=3)
    ax.plot(mon[thin], drift.loc[thin, "jsd_vs_first_third"], "o", mfc="white", mec=pal.LEAK,
            ms=3.4, zorder=3)
    split = pd.Timestamp(stat["split_date"]).tz_localize(None)
    ax.axvline(split, color=pal.SLATE, lw=0.7, ls=(0, (3, 2)))
    ax.text(split, ax.get_ylim()[1] * 0.97, " split", fontsize=6.0, color=pal.SLATE, va="top")
    ax.set_ylabel("JSD of type mix vs first third")
    ax.set_title("(b) Monthly divergence", loc="left", color=pal.INK)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6.0)

    # (c) outcome-label positive rate by month
    ax = axes[2]
    ax.plot(mon, 100 * drift["positive_rate_window_B"], "-", color=pal.INK, lw=1.3, zorder=2)
    ax.plot(mon[~thin], 100 * drift.loc[~thin, "positive_rate_window_B"], "s", color=pal.INK, ms=3.0, zorder=3)
    ax.plot(mon[thin], 100 * drift.loc[thin, "positive_rate_window_B"], "s", mfc="white", mec=pal.INK,
            ms=3.0, zorder=3)
    ax.set_ylabel("outcome-labelled positives (%)")
    ax.set_title("(c) Label rate by month", loc="left", color=pal.INK)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6.0)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=2.5, width=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_drift.pdf"))
    plt.close(fig)
    print("wrote fig6 (leakage drift)")


if __name__ == "__main__":
    if os.path.exists(os.path.join(C.RES, "fig4_h2_transfer_raw.csv")):
        fig4()
    if os.path.exists(os.path.join(C.RES, "fig5_durability_raw.csv")):
        fig5()
    if os.path.exists(os.path.join(C.RES, "h3c_stationarity.json")) and \
            os.path.exists(os.path.join(C.RES, "h3c_process_drift.csv")):
        fig6()
