#!/usr/bin/env python3
"""FIG 3 — Provenance per-record write latency (R2.5), log scale.

Local mechanisms from results/provenance_comparison.csv (N=100 row).
PoA2 bars from RAW results/scalability_benchmark.csv (N=100): strategy_a =
individual commit, strategy_b = Merkle-batch. (The report's 21.2 ms was a reused
constant; the raw CSV says 19.66 ms — we use the raw value.) Gap multipliers are
computed from the loaded values, never typed.
"""
import csv, os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import save, OKABE_ITO, COL_WIDTH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")


def row_at(path, n):
    for r in csv.DictReader(open(path)):
        if int(r["N"]) == n:
            return r
    raise SystemExit(f"N={n} not found in {path}")


def main():
    prov = row_at(os.path.join(RES, "provenance_comparison.csv"), 100)
    scal = row_at(os.path.join(RES, "scalability_benchmark.csv"), 100)
    print("FIG3 sources: results/provenance_comparison.csv (N=100),",
          "results/scalability_benchmark.csv (N=100)")

    # (label, ms, is_onchain)
    items = [
        ("Merkle root build", float(prov["merkle_root_ms_per_rec"]), False),
        ("IPFS CID (content address)", float(prov["cid_write_ms_per_rec"]), False),
        ("Ed25519 signed log", float(prov["ed25519_write_ms_per_rec"]), False),
        ("PoA$^2$ Merkle-batch", float(scal["strategy_b_per_record_ms"]), True),
        ("PoA$^2$ individual commit", float(scal["strategy_a_per_record_ms"]), True),
    ]
    for lbl, ms, _ in items:
        print(f"  {lbl}: {ms:.4f} ms/record")
    # gaps computed from loaded values
    ed = items[2][1]; cid = items[1][1]; poa = items[4][1]
    g_ed = poa / ed; g_cid = poa / cid
    print(f"  gap PoA2-individual / Ed25519 = {g_ed:,.0f}x ; / CID = {g_cid:,.0f}x")

    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    y = range(len(items))

    # Neutral, honest cost chart: a single colour for every mechanism (trust
    # differences are carried by Table A, NOT implied here by colour).
    NEUTRAL = "#7F7F7F"
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.9))
    for yi, v in zip(y, vals):
        ax.barh(yi, v, color=NEUTRAL, edgecolor="black", linewidth=0.5)
        ax.text(v * 1.4, yi, f"{v:.3g} ms" if v < 1 else f"{v:,.0f} ms",
                va="center", ha="left", fontsize=7)
    ax.set_xscale("log")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.set_xlabel("Per-record write latency (ms, log scale)")
    ax.set_xlim(4e-4, 1e5)
    ax.invert_yaxis()
    # annotate the order-of-magnitude cost gap (computed, not typed); this is a
    # cost statement, not a trust one -- placed in the empty upper-right band.
    ax.text(6.0, 0.62,
            f"PoA$^2$ individual commit is\n~{g_ed:,.0f}$\\times$ the Ed25519 log,\n"
            f"~{g_cid:,.0f}$\\times$ the IPFS CID",
            fontsize=7, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#7F7F7F", lw=0.6))
    pdf, png = save(fig, "fig3_provenance_cost")
    print("FIG3 output:", pdf, png)


if __name__ == "__main__":
    main()
