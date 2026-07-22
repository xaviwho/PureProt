#!/usr/bin/env python3
"""FIG 2 — Real-device edge throughput (R2.3). Grouped bars from raw edge JSONs.
Also emits the companion LaTeX table. No numbers typed; all from results/edge/*.json.
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import save, OKABE_ITO, COL_WIDTH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDGE = os.path.join(ROOT, "results", "edge")
TIERS = [("x86 server", "edge_x86.json", OKABE_ITO["blue"], "//"),
         ("Jetson Orin Nano", "edge_jetson.json", OKABE_ITO["orange"], "\\\\"),
         ("Raspberry Pi 4", "edge_pi4.json", OKABE_ITO["green"], "xx")]


def main():
    d = {name: json.load(open(os.path.join(EDGE, f))) for name, f, _, _ in TIERS}
    batch = [r["N"] for r in d["x86 server"]["results"]]
    print("FIG2 sources:", [f for _, f, _, _ in TIERS], "| batch sizes:", batch)

    # throughput_cpm per tier per batch (raw field)
    cpm = {name: [r["throughput_cpm"] for r in d[name]["results"]] for name, *_ in TIERS}
    for name in cpm:
        print(f"  {name}: throughput_cpm per N = {cpm[name]}")

    x = np.arange(len(batch)); w = 0.26
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.7))
    for i, (name, _, color, hatch) in enumerate(TIERS):
        bars = ax.bar(x + (i - 1) * w, cpm[name], w, label=name, color=color,
                      edgecolor="black", linewidth=0.4, hatch=hatch)
    ax.set_xticks(x); ax.set_xticklabels(batch)
    ax.set_xlabel("Batch size (compounds)")
    ax.set_ylabel("Throughput (compounds/min)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
    ax.margins(y=0.10)
    pdf, png = save(fig, "fig2_edge_throughput")
    print("FIG2 output:", pdf, png)

    # LaTeX table: tier, cores, throughput (mean cpm), vs server, peak RSS @Nmax, infer share
    def mean(v): return sum(v) / len(v)
    rows = []
    server_cpm = mean(cpm["x86 server"])
    for name, f, _, _ in TIERS:
        res = d[name]["results"]; man = d[name]["manifest"]
        cpm_mean = mean([r["throughput_cpm"] for r in res])
        peak = max(r["peak_rss_mb"] for r in res)
        infer_share = mean([100 * r["infer_ms"] / r["total_ms"] for r in res])
        rows.append((name, man["cpu_count"], cpm_mean, server_cpm / cpm_mean, peak, infer_share))
        print(f"  TABLE {name}: cores={man['cpu_count']} cpm={cpm_mean:.0f} "
              f"vs_server={server_cpm/cpm_mean:.2f} peak={peak:.0f}MB infer={infer_share:.0f}%")
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Tier & Cores & Throughput & vs.\ server & Peak RSS & Inference \\",
           r" & & (cpd/min) & ($\times$ slower) & (MB) & share (\%) \\", r"\midrule"]
    for name, cores, c, ratio, peak, infer in rows:
        tex.append(f"{name} & {cores} & {c:,.0f} & {ratio:.2f} & {peak:.0f} & {infer:.0f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    p = os.path.join(ROOT, "figures", "tables", "table_fig2_edge.tex")
    open(p, "w").write("\n".join(tex)); print("FIG2 LaTeX:", p)


if __name__ == "__main__":
    main()
