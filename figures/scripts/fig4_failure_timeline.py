#!/usr/bin/env python3
"""FIG 4 — Validator failure-injection + gap-condition timeline (R1.2).

Panel (a): fault envelope from failure_timeseries.csv (+ failure_injection.json).
Panel (b): gap-condition controller from gap_controller_timeseries.csv (+ .json).
All heights, rates, event times, and latencies are read from the files.
"""
import csv, json, os, sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import save, OKABE_ITO, DBL_WIDTH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A4 = os.path.join(ROOT, "results", "a4_testnet")
NODE2 = "0xe2c299d8a098080a756bcac9e56c99b2ed4ece2f"   # the killed active signer


def panel_a(ax):
    rows = list(csv.DictReader(open(os.path.join(A4, "failure_timeseries.csv"))))
    inj = json.load(open(os.path.join(A4, "failure_injection.json")))
    order = ["baseline_4of4", "kill1_3of4", "kill2_2of4", "recovered_4of4"]
    plabel = {"baseline_4of4": "4/4", "kill1_3of4": "3/4", "kill2_2of4": "2/4  (STALL)", "recovered_4of4": "4/4"}
    # concatenate per-phase t_s into a continuous axis (t_s resets each phase)
    offset, bounds, t, h = 0.0, [], [], []
    for ph in order:
        pr = [r for r in rows if r["phase"] == ph]
        maxt = max(float(r["t_s"]) for r in pr)
        bounds.append((ph, offset, offset + maxt))
        for r in pr:
            t.append(offset + float(r["t_s"])); h.append(int(r["max_height"]))
        offset += maxt + 2.0   # ~2s gap for the kill/restart action
    ax.plot(t, h, color=OKABE_ITO["black"], lw=1.3, marker="o", ms=2.2)
    print("PANEL a: phases", [(b[0], round(b[1],1), round(b[2],1)) for b in bounds])
    print("PANEL a: recovery", inj["recovery"], "| rates",
          {p["phase"]: p["block_rate_per_s"] for p in inj["phases"]})
    ymin = min(h)
    for i, (ph, a, b) in enumerate(bounds):
        ax.text((a + b) / 2, max(h) + 1.5, plabel[ph], ha="center", va="bottom",
                fontsize=7, fontweight="bold",
                color=OKABE_ITO["vermillion"] if ph == "kill2_2of4" else OKABE_ITO["black"])
        if i > 0:
            ev = {"kill1_3of4": "kill v2", "kill2_2of4": "kill v3", "recovered_4of4": "restart"}[ph]
            ax.axvline(a - 1.0, color=OKABE_ITO["grey"], ls="--", lw=0.8)
            ax.annotate(ev, xy=(a - 1.0, ymin), xytext=(a - 1.0, ymin - 3.0),
                        ha="center", fontsize=6.5, color=OKABE_ITO["blue"])
    # recovery annotation from json
    rec = inj["recovery"]
    rb = bounds[3][1]
    ax.annotate(f"recovery {rec['recovery_time_s']} s",
                xy=(rb, rec["height_after"]), xytext=(rb - 14, max(h) - 4),
                fontsize=6.5, color=OKABE_ITO["green"],
                arrowprops=dict(arrowstyle="->", color=OKABE_ITO["green"], lw=0.7))
    ax.set_xlabel("(a) static 4-signer set: elapsed time (s, phases concatenated)")
    ax.set_ylabel("Block height")
    ax.set_ylim(ymin - 5, max(h) + 5)


def panel_b(ax):
    rows = list(csv.DictReader(open(os.path.join(A4, "gap_controller_timeseries.csv"))))
    gc = json.load(open(os.path.join(A4, "gap_controller.json")))
    t = [float(r["t_s"]) for r in rows]
    h = [int(r["max_height"]) for r in rows]
    # node2 reliability (cell JSON has commas escaped as ';')
    r2 = []
    for r in rows:
        try:
            d = json.loads(r["reliabilities"].replace(";", ","))
            r2.append(d.get(NODE2))
        except Exception:
            r2.append(None)
    h_line, = ax.plot(t, h, color=OKABE_ITO["black"], lw=1.3, marker="o", ms=2.2,
                      label="Block height")
    ax.set_ylabel("Block height")
    ax.set_xlabel("(b) with gap-condition controller: elapsed time (s)")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    tt = [(x, y) for x, y in zip(t, r2) if y is not None]
    r_line, = ax2.plot([x for x, _ in tt], [y for _, y in tt], color=OKABE_ITO["vermillion"],
                       lw=1.2, ls="--", marker="s", ms=2.5, label="Reliability $R$(v2)")
    ax2.set_ylabel("Reliability $R$(v2)", color=OKABE_ITO["vermillion"])
    ax2.tick_params(axis="y", colors=OKABE_ITO["vermillion"]); ax2.set_ylim(-0.05, 1.15)
    # De-crowded: only compact event markers here; the detection/promotion
    # latencies and the "no-stall" narrative are stated in the caption.
    t_fault = next(e["t_s"] for e in gc["events"] if "node2 killed" in e["event"])
    t_gap = next(e["t_s"] for e in gc["events"] if "GAP CONDITION" in e["event"])
    t_seal = next(e["t_s"] for e in gc["events"] if "sealing" in e["event"])
    t_f2 = next(e["t_s"] for e in gc["events"] if "node3 killed" in e["event"])
    print("PANEL b: detect", gc["gap_detection_latency_s"], "promote", gc["promotion_latency_s"],
          "| payoff stalled=", gc["payoff_second_fault"]["stalled"])
    marks = [(t_fault, "v2\nkilled"), (t_gap, "gap"), (t_seal, "standby\nsealing"),
             (t_f2, "v3\nkilled")]
    ax.set_ylim(min(h) - 3, max(h) + 8)
    ytop = max(h) + 8
    for x, lab in marks:
        ax.axvline(x, color=OKABE_ITO["grey"], ls=":", lw=0.9)
        ax.text(x, ytop, lab, fontsize=6.5, color=OKABE_ITO["black"],
                ha="center", va="top")
    # single legend from the two DATA lines only (exclude event axvlines)
    ax.legend([h_line, r_line], [h_line.get_label(), r_line.get_label()],
              loc="lower right", frameon=False, fontsize=6.8)


def main():
    print("FIG4 sources: results/a4_testnet/{failure_timeseries.csv,failure_injection.json,"
          "gap_controller_timeseries.csv,gap_controller.json}")
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(DBL_WIDTH, 4.4))
    panel_a(axa); panel_b(axb)
    pdf, png = save(fig, "fig4_failure_timeline")
    print("FIG4 output:", pdf, png)


if __name__ == "__main__":
    main()
