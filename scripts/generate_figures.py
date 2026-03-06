"""
Generate all paper figures and tables for PureProtX revision.
Outputs to experiments/paper_results/figures/

One panel per figure. Large, legible fonts throughout.

  Table B1  - 10-target benchmark panel (CSV + LaTeX)
  Fig   C1  - 10-target performance heatmap (R2/RMSE/AUC-ROC/AUC-PR)
  Table C2  - Early-recognition results by method + target (CSV + LaTeX)
  Fig   D1  - Per-target alpha bar chart with LOTO generalisation overlay
  Fig   E1  - Novel scaffold fraction improvements (alpha<1 targets)
  Table F2  - DUD-E transfer results + alpha (CSV + LaTeX)
  Table G1  - Blockchain provenance comparison (CSV + LaTeX)
  Fig   G2  - Blockchain pipeline Merkle audit + tamper detection
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..")
RESULTS_JSON = os.path.join(ROOT, "experiments", "paper_results", "revised_results.json")
OUT_DIR = os.path.join(ROOT, "experiments", "paper_results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS_JSON) as fh:
    DATA = json.load(fh)
EXPS = DATA["experiments"]

# ---------------------------------------------------------------------------
# Target metadata
# ---------------------------------------------------------------------------
TARGETS = [
    ("CHEMBL243",  "HIV-1 Protease",              "Protease"),
    ("CHEMBL247",  "HIV-1 RT",                    "Reverse Transcriptase"),
    ("CHEMBL279",  "VEGFR2",                      "Kinase"),
    ("CHEMBL3471", "Tankyrase-2",                 "Transferase"),
    ("CHEMBL2487", "Amyloid-beta APP",            "Membrane protein"),
    ("CHEMBL251",  "Adenosine A2a",               "GPCR"),
    ("CHEMBL217",  "Dopamine D2",                 "GPCR"),
    ("CHEMBL1862", "Estrogen R.alpha",            "Nuclear receptor"),
    ("CHEMBL4005", "PPARgamma",                   "Nuclear receptor"),
    ("CHEMBL240",  "hERG",                        "Ion channel"),
]
TARGET_IDS   = [t[0] for t in TARGETS]
TARGET_NAMES = [t[1] for t in TARGETS]
TARGET_FAM   = [t[2] for t in TARGETS]

SHORT = {t[0]: t[1] for t in TARGETS}

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
BASE_FONT   = 13
TITLE_FONT  = 15
LABEL_FONT  = 13
TICK_FONT   = 11
ANNOT_FONT  = 10
SAVE_DPI    = 300

BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
ORANGE = "#F4A742"
GRAY   = "#888888"
DGRAY  = "#333333"
LIGHT  = "#EEEEEE"

plt.rcParams.update({
    "font.size":        BASE_FONT,
    "axes.titlesize":   TITLE_FONT,
    "axes.labelsize":   LABEL_FONT,
    "xtick.labelsize":  TICK_FONT,
    "ytick.labelsize":  TICK_FONT,
    "legend.fontsize":  TICK_FONT,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      SAVE_DPI,
    "font.family":      "DejaVu Sans",
})

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# TABLE B1 — 10-target benchmark panel
# ===========================================================================
def table_b1():
    exp1 = EXPS["exp1_model_training"]
    rows = []
    for tid, name, fam in TARGETS:
        e = exp1[tid]
        rows.append((tid, name, fam,
                     e["n_total"], e["n_train"], e["n_val"], e["n_test"],
                     e["docking_method"]))

    csv_path = os.path.join(OUT_DIR, "table_B1_benchmark_panel.csv")
    with open(csv_path, "w") as f:
        f.write("ChEMBL,Target,Family,N,Train,Val,Test,Docking\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    tex_path = os.path.join(OUT_DIR, "table_B1_benchmark_panel.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n"
                "\\caption{Table B1: 10-target benchmark panel.}\n"
                "\\label{tab:benchmark_panel}\n"
                "\\centering\\small\n"
                "\\begin{tabular}{llcrrrrl}\n"
                "\\toprule\n"
                "ChEMBL & Target & Family & $N$ & Train & Val & Test & Docking \\\\\n"
                "\\midrule\n")
        for r in rows:
            f.write(f"{r[0]} & {r[1]} & {r[2]} & {r[3]:,} & {r[4]:,} & "
                    f"{r[5]:,} & {r[6]:,} & {r[7].capitalize()} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  [B1] {csv_path}")
    print(f"  [B1] {tex_path}")


# ===========================================================================
# FIG C1 — Performance heatmap (one clean panel)
# ===========================================================================
def fig_c1():
    exp1 = EXPS["exp1_model_training"]
    metrics    = ["R$^2$", "RMSE", "AUC-ROC", "AUC-PR"]
    lower_better = [False, True, False, False]

    matrix = []
    for tid in TARGET_IDS:
        e  = exp1[tid]
        rc = e["regression_metrics"]["consensus"]
        cc = e["classification_metrics"]["consensus_clf"]
        matrix.append([
            rc["test_r2"],
            rc["test_rmse"],
            cc["test_auc_roc"],
            cc["test_auc_pr"],
        ])
    mat = np.array(matrix).T            # shape (4 metrics, 10 targets)

    # Normalise per row for colour; invert RMSE so green = good
    mat_norm = np.zeros_like(mat)
    for i in range(len(metrics)):
        lo, hi = mat[i].min(), mat[i].max()
        n = (mat[i] - lo) / (hi - lo) if hi > lo else np.full_like(mat[i], 0.5)
        mat_norm[i] = 1.0 - n if lower_better[i] else n

    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(mat_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(TARGET_IDS)))
    ax.set_xticklabels([SHORT[t] for t in TARGET_IDS],
                       rotation=30, ha="right", fontsize=TICK_FONT + 1)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=TICK_FONT + 1)

    for i in range(len(metrics)):
        for j in range(len(TARGET_IDS)):
            val  = mat[i, j]
            norm = mat_norm[i, j]
            text_color = "white" if (norm < 0.28 or norm > 0.80) else "black"
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=ANNOT_FONT + 1,
                    color=text_color,
                    fontweight="bold")

    ax.set_title("10-Target Consensus Model Performance (test set)",
                 pad=10, fontsize=TITLE_FONT + 1)
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("Relative performance\n(green = best in row)",
                 fontsize=TICK_FONT)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"], fontsize=TICK_FONT)
    fig.tight_layout()
    save(fig, "fig_C1_performance_heatmap.png")


# ===========================================================================
# TABLE C2 — Early-recognition results by method + target
# ===========================================================================
def table_c2():
    exp2 = EXPS["exp2_enrichment_metrics"]
    rows = []
    for tid in TARGET_IDS:
        e = exp2[tid]
        alpha = e["optimal_alpha"]
        for key, label in [("regression", "Regression"),
                            ("classification", "Classification"),
                            ("hybrid", f"Hybrid(a={alpha:.2f})")]:
            m = e[key]
            rows.append((SHORT[tid], label,
                         round(m["ef1"], 2),
                         round(m["ef5"], 2),
                         round(m["ef10"], 2),
                         round(m["bedroc_20"], 3),
                         round(m["auc_roc"], 3)))

    csv_path = os.path.join(OUT_DIR, "table_C2_enrichment_results.csv")
    with open(csv_path, "w") as f:
        f.write("Target,Method,EF@1%,EF@5%,EF@10%,BEDROC,AUC-ROC\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    tex_path = os.path.join(OUT_DIR, "table_C2_enrichment_results.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n"
                "\\caption{Table C2: Early-recognition metrics on ChEMBL test sets. "
                "EF = enrichment factor; BEDROC computed at $\\alpha{=}20$.}\n"
                "\\label{tab:enrichment}\n"
                "\\centering\\small\n"
                "\\begin{tabular}{llrrrrr}\n"
                "\\toprule\n"
                "Target & Method & EF@1\\% & EF@5\\% & EF@10\\% & BEDROC & AUC-ROC \\\\\n"
                "\\midrule\n")
        prev = None
        for r in rows:
            if r[0] != prev and prev is not None:
                f.write("\\midrule\n")
            prev = r[0]
            f.write(f"{r[0]} & {r[1]} & {r[2]:.2f} & {r[3]:.2f} & "
                    f"{r[4]:.2f} & {r[5]:.3f} & {r[6]:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  [C2] {csv_path}")
    print(f"  [C2] {tex_path}")


# ===========================================================================
# FIG D1 — Per-target alpha with LOTO overlay (single panel)
# ===========================================================================
def fig_d1():
    exp3      = EXPS["exp3_alpha_optimization"]
    per_t     = exp3["per_target_alphas"]
    loto      = exp3["loto"]
    loto_mean = loto["mean_alpha"]
    loto_std  = loto["std_alpha"]

    tids    = [t for t in TARGET_IDS if t in per_t]
    alphas  = [per_t[t]["alpha"] for t in tids]
    labels  = [SHORT[t] for t in tids]
    colors  = [RED if a < 1.0 else BLUE for a in alphas]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(tids))
    bars = ax.bar(x, alphas, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.55, zorder=3)

    # LOTO band and line
    ax.fill_between([-0.6, len(tids) - 0.4],
                    loto_mean - loto_std, loto_mean + loto_std,
                    color=GREEN, alpha=0.15, zorder=1,
                    label=f"LOTO mean ± SD  ({loto_mean:.3f} ± {loto_std:.3f})")
    ax.axhline(loto_mean, color=GREEN, lw=2.0, ls="-.",
               zorder=2)

    # Per-target mean line
    pt_mean = np.mean(alphas)
    ax.axhline(pt_mean, color=ORANGE, lw=2.0, ls="--",
               label=f"Per-target mean ({pt_mean:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_FONT + 1)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Optimal alpha  (AI weight in hybrid score)", fontsize=LABEL_FONT)
    ax.set_title("Alpha Optimisation: Per-target optimal alpha with LOTO generalisation",
                 fontsize=TITLE_FONT, pad=10)

    # Value annotations above bars
    for i, v in enumerate(alphas):
        ax.text(i, v + 0.03, f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=ANNOT_FONT + 1, color=DGRAY, fontweight="bold")

    # Legend patches
    blue_p  = mpatches.Patch(color=BLUE,  label="alpha = 1.0  (AI only)")
    red_p   = mpatches.Patch(color=RED,   label="alpha < 1.0  (docking contributes)")
    loto_l  = plt.Line2D([0],[0], color=GREEN,  lw=2.0, ls="-.",
                          label=f"LOTO mean ± SD  ({loto_mean:.3f} ± {loto_std:.3f})")
    mean_l  = plt.Line2D([0],[0], color=ORANGE, lw=2.0, ls="--",
                          label=f"Per-target mean ({pt_mean:.3f})")
    ax.legend(handles=[blue_p, red_p, loto_l, mean_l],
              loc="upper left", framealpha=0.9, fontsize=TICK_FONT)

    fig.tight_layout()
    save(fig, "fig_D1_alpha_optimization.png")


# ===========================================================================
# FIG E1 — Novel scaffold fraction (alpha<1 targets, single panel)
# ===========================================================================
def fig_e1():
    exp6  = EXPS["exp6_scaffold_diversity"]
    per_t = EXPS["exp3_alpha_optimization"]["per_target_alphas"]

    a_targets = [(t, per_t[t]["alpha"])
                 for t in TARGET_IDS
                 if t in per_t and per_t[t]["alpha"] < 1.0]
    a_targets.sort(key=lambda x: x[1])   # sort by alpha ascending

    labels = [f"{SHORT[t]}\n(α={a:.2f})" for t, a in a_targets]
    tids   = [t for t, _ in a_targets]

    reg = [exp6[t]["regression"]["novel_scaffold_fraction"] * 100  for t in tids]
    clf = [exp6[t]["classification"]["novel_scaffold_fraction"] * 100 for t in tids]
    hyb = [exp6[t]["hybrid"]["novel_scaffold_fraction"] * 100      for t in tids]

    x = np.arange(len(tids))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))

    b_reg = ax.bar(x - w,     reg, width=w, color=BLUE,   label="Regression",
                   edgecolor="white", linewidth=0.6, zorder=3)
    b_clf = ax.bar(x,         clf, width=w, color=ORANGE, label="Classification",
                   edgecolor="white", linewidth=0.6, zorder=3)
    b_hyb = ax.bar(x + w,     hyb, width=w, color=GREEN,  label="Hybrid",
                   edgecolor="white", linewidth=0.6, zorder=3)

    # Annotate hybrid vs regression delta
    for i in range(len(tids)):
        diff = hyb[i] - reg[i]
        sign = "+" if diff >= 0 else ""
        ypos = max(reg[i], clf[i], hyb[i]) + 0.8
        ax.text(x[i], ypos, f"{sign}{diff:.1f}%",
                ha="center", va="bottom",
                fontsize=ANNOT_FONT, color=DGRAY, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FONT + 1)
    ax.set_ylabel("Novel scaffold fraction (%)", fontsize=LABEL_FONT)
    ax.set_title("Novel Scaffold Fraction in Top-10% Predictions\n"
                 "(targets where docking contributes, alpha < 1.0)",
                 fontsize=TITLE_FONT, pad=10)
    ax.set_ylim(0, max(reg + clf + hyb) * 1.22)
    ax.legend(fontsize=TICK_FONT, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", color=LIGHT, zorder=0)

    # Delta annotation explanation
    ax.text(0.99, 0.97, "Annotated delta: Hybrid vs Regression",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=ANNOT_FONT - 1, color=GRAY, style="italic")

    fig.tight_layout()
    save(fig, "fig_E1_scaffold_diversity.png")


# ===========================================================================
# TABLE F2 — DUD-E transfer results + alpha
# ===========================================================================
def fig_f2_table():
    exp7 = EXPS["exp7_dude_evaluation"]
    DUDE_META = {
        "aa2ar": ("CHEMBL251",  "Adenosine A2a",    "GPCR"),
        "esr1":  ("CHEMBL1862", "Estrogen R.alpha",  "Nuclear receptor"),
        "hivpr": ("CHEMBL243",  "HIV-1 Protease",    "Protease"),
        "pparg": ("CHEMBL4005", "PPARgamma",          "Nuclear receptor"),
        "vgfr2": ("CHEMBL279",  "VEGFR2",             "Kinase"),
    }
    ORDER = ["hivpr", "vgfr2", "aa2ar", "esr1", "pparg"]

    rows = []
    for dk in ORDER:
        e = exp7[dk]
        _, name, fam = DUDE_META[dk]
        alpha = e.get("optimal_alpha", float("nan"))
        h_b   = e.get("hybrid", {}).get("bedroc_20", float("nan"))
        rows.append((dk.upper(), name, fam,
                     e["n_actives_valid"], e["n_decoys"],
                     alpha,
                     e["regression"]["bedroc_20"],
                     e["classification"]["bedroc_20"],
                     h_b,
                     e["regression"]["auc_roc"]))

    csv_path = os.path.join(OUT_DIR, "table_F2_dude_transfer.csv")
    with open(csv_path, "w") as f:
        f.write("DUD-E,Target,Family,N_actives,N_decoys,alpha,"
                "Reg_BEDROC,Clf_BEDROC,Hyb_BEDROC,Reg_AUC\n")
        for r in rows:
            h = f"{r[8]:.3f}" if not math.isnan(r[8]) else "N/A"
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.2f},"
                    f"{r[6]:.3f},{r[7]:.3f},{h},{r[9]:.3f}\n")

    tex_path = os.path.join(OUT_DIR, "table_F2_dude_transfer.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n"
                "\\caption{Table F2: DUD-E external transfer benchmark. "
                "BEDROC$_{\\alpha=20}$ shown. alpha = ChEMBL-optimised docking weight.}\n"
                "\\label{tab:dude}\n"
                "\\centering\\small\n"
                "\\begin{tabular}{llcrrrrrrr}\n"
                "\\toprule\n"
                "DUD-E & Target & Family & $N_{act}$ & $N_{dec}$ & "
                "$\\alpha$ & Reg BEDROC & Clf BEDROC & Hyb BEDROC & Reg AUC \\\\\n"
                "\\midrule\n")
        for r in rows:
            h = f"{r[8]:.3f}" if not math.isnan(r[8]) else "---"
            f.write(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & "
                    f"{r[5]:.2f} & {r[6]:.3f} & {r[7]:.3f} & {h} & {r[9]:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  [F2] {csv_path}")
    print(f"  [F2] {tex_path}")


# ===========================================================================
# TABLE G1 — Blockchain provenance comparison
# ===========================================================================
def table_g1():
    rows = [
        ("Immutable record",    "No",    "No*",   "Yes"),
        ("Tamper detectable",   "No",    "Yes*",  "Yes"),
        ("Merkle audit trail",  "No",    "No",    "Yes"),
        ("Replay protection",   "No",    "No",    "Yes"),
        ("Cost per record",     "Free",  "Free",  "Free (zero-gas)"),
        ("Offline mode",        "N/A",   "Yes",   "Yes"),
        ("Multi-stage hashing", "No",    "No",    "Yes"),
    ]
    csv_path = os.path.join(OUT_DIR, "table_G1_blockchain_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("Feature,No Provenance,File Hash Only,PureChain\n")
        for r in rows:
            f.write(",".join(r) + "\n")

    tex_path = os.path.join(OUT_DIR, "table_G1_blockchain_comparison.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n"
                "\\caption{Table G1: Provenance audit method comparison. "
                "*File-based hashes can be overwritten if the attacker controls the filesystem.}\n"
                "\\label{tab:provenance}\n"
                "\\centering\\small\n"
                "\\begin{tabular}{lccc}\n"
                "\\toprule\n"
                "Feature & No provenance & File hash only & PureChain \\\\\n"
                "\\midrule\n")
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  [G1] {csv_path}")
    print(f"  [G1] {tex_path}")


# ===========================================================================
# FIG G2 — Blockchain tamper detection (single, clean panel)
# ===========================================================================
def fig_g2():
    exp5 = EXPS["exp5_tamper_detection"]
    td   = exp5["tamper_detection"]
    mt   = exp5["merkle_tree"]
    bc   = exp5["blockchain"]

    orig_h   = td["original_hash"][:20]   + "..."
    tamper_h = td["tampered_hash"][:20]   + "..."
    root     = mt["merkle_root"][:20]     + "..."
    tamper_tx = bc["tamper_tx"]["tx_hash"][:20] + "..."
    blk       = bc["tamper_tx"]["block_number"]
    stages    = ["Fetch data", "Train model", "Dock ligands", "Score consensus"]
    s_hashes  = [mt["stage_hashes"][k][:14] + "..."
                 for k in ["fetch", "train", "dock", "score"]]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    # ---------- helper: rounded box ----------
    def rbox(ax, cx, cy, w, h, text, fc, ec, fs=11, bold=False, mono=False):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
        ax.add_patch(rect)
        family = "monospace" if mono else "DejaVu Sans"
        weight = "bold" if bold else "normal"
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, family=family,
                color=DGRAY, zorder=4, wrap=False)

    def arrow(ax, x1, y1, x2, y2, color=DGRAY, lw=1.8, ls="-", label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, linestyle=ls),
                    zorder=2)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=9, color=color, va="center")

    # ---- COLUMN 1: Pipeline stages (left) ----
    ax.text(2.5, 9.5, "4-Stage Pipeline", ha="center", fontsize=13,
            fontweight="bold", color=BLUE)
    stage_ys = [8.5, 7.0, 5.5, 4.0]
    stage_fc = "#DDEEFF"
    hash_fc  = "#FFF8E1"
    for i, (s, h, y) in enumerate(zip(stages, s_hashes, stage_ys)):
        # stage box
        rbox(ax, 1.8, y, 3.0, 0.65, s, stage_fc, BLUE, fs=11, bold=True)
        # arrow to hash box
        arrow(ax, 3.35, y, 4.4, y, color=BLUE)
        # hash box
        rbox(ax, 5.5, y, 2.1, 0.60, f"H{i+1}: {h}", hash_fc, ORANGE, fs=9, mono=True)
        if i < len(stage_ys) - 1:
            arrow(ax, 5.5, y - 0.32, 5.5, stage_ys[i+1] + 0.32, color=ORANGE, lw=1.4)

    # Merkle root box
    arrow(ax, 5.5, 3.68, 5.5, 3.05, color=GREEN, lw=2.0)
    rbox(ax, 5.5, 2.65, 2.4, 0.65,
         f"Merkle root:\n{root}", "#E8FFE8", GREEN, fs=9, mono=True)
    # to chain
    arrow(ax, 6.72, 2.65, 7.7, 2.65, color=DGRAY, lw=2.0)
    rbox(ax, 8.7, 2.65, 1.8, 0.65,
         "PureChain\n(on-chain)", "#F0E8FF", "#8844CC", fs=10, bold=True)

    # ---- COLUMN 2: Tamper detection (right) ----
    ax.text(11.3, 9.5, "Tamper Detection", ha="center", fontsize=13,
            fontweight="bold", color=RED)

    # Original result
    rbox(ax, 11.3, 8.5, 3.2, 0.65,
         "Original screening result", "#DDEEFF", BLUE, fs=10, bold=True)
    arrow(ax, 11.3, 8.17, 11.3, 7.65, color=BLUE)
    rbox(ax, 11.3, 7.3, 3.2, 0.65,
         f"Hash H1:\n{orig_h}", "#E8FFE8", GREEN, fs=9, mono=True)

    # Store on-chain
    arrow(ax, 11.3, 6.97, 11.3, 6.35, color=DGRAY, lw=1.6)
    rbox(ax, 11.3, 6.0, 3.4, 0.65,
         f"Stored on-chain\ntx: {tamper_tx}\nblock: {blk:,}", "#F0E8FF",
         "#8844CC", fs=8, mono=True)

    # Adversary tampers
    arrow(ax, 11.3, 5.67, 11.3, 4.95, color=RED, lw=2.0, ls="--")
    ax.text(11.7, 5.35, "adversary\ntampers", fontsize=9, color=RED, va="center")
    rbox(ax, 11.3, 4.55, 3.2, 0.65,
         f"New Hash H2:\n{tamper_h}", "#FFE8E8", RED, fs=9, mono=True)

    # Mismatch
    arrow(ax, 11.3, 4.22, 11.3, 3.5, color=RED, lw=2.0)
    rbox(ax, 11.3, 3.12, 3.5, 0.70,
         "H2  !=  H1\nTAMPER DETECTED", "#FFD0D0", RED, fs=12, bold=True)

    # Divider
    ax.plot([7.15, 7.15], [0.3, 9.8], color=GRAY, lw=1.0, ls=":", zorder=1)

    ax.set_title(
        "Fig G2: PureChain Blockchain Provenance — "
        "Merkle Audit Pipeline and Tamper Detection",
        fontsize=TITLE_FONT, pad=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig_G2_blockchain.png")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Generating PureProtX paper figures and tables ...")
    print()

    print("[Table B1]  10-target benchmark panel")
    table_b1()

    print("[Fig   C1]  Performance heatmap")
    fig_c1()

    print("[Table C2]  Early-recognition results")
    table_c2()

    print("[Fig   D1]  Alpha optimisation")
    fig_d1()

    print("[Fig   E1]  Scaffold diversity (alpha<1 targets)")
    fig_e1()

    print("[Table F2]  DUD-E transfer results")
    fig_f2_table()

    print("[Table G1]  Blockchain provenance comparison")
    table_g1()

    print("[Fig   G2]  Blockchain tamper detection")
    fig_g2()

    print()
    print("All outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
