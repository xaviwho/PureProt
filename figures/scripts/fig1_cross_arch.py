#!/usr/bin/env python3
"""FIG 1 — Cross-architecture determinism matrix (R2.2/R2.4).

Reads the RAW per-device harness JSON (canonical_hash per model x thread at the
pinned ORT 1.18.0) and computes MATCH/DIFFER of each aarch64 device vs the x86
reference. No numbers are typed; everything is derived from the JSON.
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import save, OKABE_ITO, DBL_WIDTH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DET = os.path.join(ROOT, "results", "determinism")
SRC = {"x86": "harness_ort1.18.0.json", "Jetson": "harness_ort1.18.0_arm.json",
       "Pi 4": "harness_ort1.18.0_pi4.json"}
DEVICES = ["x86", "Jetson", "Pi 4"]
THREADS = ["threads_1", "threads_2", "threads_4"]
TLABEL = {"threads_1": "1", "threads_2": "2", "threads_4": "4"}
PRETTY = {"reg_svr": "SVR (reg)", "reg_random_forest": "RandomForest (reg)",
          "reg_gradient_boosting": "GradBoosting (reg)", "clf_svc": "SVC (clf)",
          "clf_rf_clf": "RandomForest (clf)", "clf_gb_clf": "GradBoosting (clf)",
          "scaler": "StandardScaler"}
ORDER = ["reg_svr", "reg_random_forest", "reg_gradient_boosting",
         "clf_svc", "clf_rf_clf", "clf_gb_clf", "scaler"]


def main():
    data = {dev: json.load(open(os.path.join(DET, f)))["results"] for dev, f in SRC.items()}
    models = [m for m in ORDER if m in data["x86"]]
    ncol = len(DEVICES) * len(THREADS)
    grid = np.ones((len(models), ncol))      # 1 = MATCH, 0 = DIFFER
    col_dev, col_thr = [], []
    print("FIG1 sources:", {d: SRC[d] for d in DEVICES})
    for ci_dev, dev in enumerate(DEVICES):
        for ti, tk in enumerate(THREADS):
            c = ci_dev * len(THREADS) + ti
            col_dev.append(dev); col_thr.append(TLABEL[tk])
            for ri, m in enumerate(models):
                ref = data["x86"][m][tk]["canonical_hash"]
                h = data[dev][m][tk]["canonical_hash"]
                grid[ri, c] = 1.0 if h == ref else 0.0
    n_match = int(grid.sum()); n_diff = int(grid.size - n_match)
    print(f"FIG1 cells: {grid.size} total, MATCH={n_match}, DIFFER={n_diff}")
    diff_models = sorted({models[r] for r, c in zip(*np.where(grid == 0))})
    print("FIG1 DIFFER models:", diff_models)

    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(DBL_WIDTH, 2.9))
    # Draw each cell as a rectangle: color + (for DIFFER) a diagonal hatch so the
    # figure survives grayscale printing without relying on any glyph font.
    for r in range(len(models)):
        for c in range(ncol):
            match = grid[r, c] == 1
            ax.add_patch(Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=OKABE_ITO["green"] if match else OKABE_ITO["vermillion"],
                edgecolor="white", linewidth=1.0,
                hatch=None if match else "////"))
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(len(models) - 0.5, -1.15)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([PRETTY[m] for m in models])
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(col_thr)
    ax.set_xlabel("intra-op threads, grouped by device (ONNX Runtime 1.18.0)")
    for d, dev in enumerate(DEVICES):
        ax.text(d * 3 + 1, -0.8, dev, ha="center", va="bottom", fontweight="bold", fontsize=8)
    for d in range(1, len(DEVICES)):
        ax.axvline(d * 3 - 0.5, color="black", lw=1.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    leg = [Patch(facecolor=OKABE_ITO["green"], edgecolor="white",
                 label="MATCH: byte-identical vs x86"),
           Patch(facecolor=OKABE_ITO["vermillion"], edgecolor="white", hatch="////",
                 label="DIFFER: hash differs vs x86")]
    ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.46), ncol=2, frameon=False)
    pdf, png = save(fig, "fig1_cross_arch")
    print("FIG1 output:", pdf, png)

    # LaTeX table of the grid
    hdr = " & ".join([f"\\multicolumn{{3}}{{c}}{{{d}}}" for d in DEVICES])
    subhdr = " & ".join(col_thr)
    lines = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
             "Model & " + hdr + r" \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             " & " + subhdr + r" \\", r"\midrule"]
    for ri, m in enumerate(models):
        cells = ["\\checkmark" if grid[ri, c] else "$\\times$" for c in range(ncol)]
        lines.append(PRETTY[m].replace("&", "\\&") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = os.path.join(ROOT, "figures", "tables", "table_fig1_cross_arch.tex")
    open(tex, "w").write("\n".join(lines))
    print("FIG1 LaTeX:", tex)
    print(f"FIG1 caption note: {n_match}/{grid.size} cells byte-identical across the "
          f"three devices; sole DIFFER = {', '.join(PRETTY[m] for m in diff_models)}.")


if __name__ == "__main__":
    main()
