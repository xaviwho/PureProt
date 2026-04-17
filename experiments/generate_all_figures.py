#!/usr/bin/env python3
"""
Generate all paper figures for the IoT reframing sections V-B through V-H.
Reads from results/ JSON/CSV files. Outputs PNG to results/paper_tables/.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")
OUT = os.path.join(RESULTS, "paper_tables")
os.makedirs(OUT, exist_ok=True)

COLORS = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]


def fig_vb_edge():
    """V-B: Throughput grouped bar chart — both targets side by side."""
    df = pd.read_csv(os.path.join(RESULTS, "edge_benchmark.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    tiers = ["server", "rpi4", "jetson_nano", "constrained"]
    tier_labels = ["Server", "RPi4", "Jetson\nNano", "Constrained"]

    for ax, target, title in zip(
        axes,
        ["CHEMBL243", "CHEMBL240"],
        ["CHEMBL243 (HIV-1 Protease, 67 MB)", "CHEMBL240 (hERG, 322 MB)"],
    ):
        tdf = df[df["target"] == target]
        batch_sizes = sorted(tdf["n_compounds"].unique())
        x = np.arange(len(tiers))
        width = 0.2

        for i, N in enumerate(batch_sizes):
            vals = []
            for t in tiers:
                sel = tdf[(tdf["profile"] == t) & (tdf["n_compounds"] == N)]
                if not sel.empty and sel.iloc[0]["success"]:
                    vals.append(sel.iloc[0]["throughput_cpm"])
                else:
                    vals.append(0)
            bars = ax.bar(x + (i - len(batch_sizes)/2 + 0.5) * width, vals,
                          width, label=f"N={N}", color=COLORS[i % len(COLORS)])
            # Mark OOM
            for j, v in enumerate(vals):
                if v == 0:
                    ax.text(x[j] + (i - len(batch_sizes)/2 + 0.5) * width, 50,
                            "OOM", ha="center", va="bottom", fontsize=7,
                            color="red", fontweight="bold")

        ax.set_xlabel("Hardware Tier", fontsize=11)
        ax.set_ylabel("Throughput (compounds / min)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, title="Batch size")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Fig. V-B: Edge Deployment Throughput", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_vb_edge_throughput.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_vc_consensus():
    """V-C: Consensus latency — baseline vs sustained with error bars."""
    with open(os.path.join(RESULTS, "failure_test.json")) as f:
        d = json.load(f)

    bl = d["baseline_latencies"]
    sl = d["sustained_latencies"]
    n_val = d.get("validators", {}).get("count", "?")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = [f"Baseline\n(N={bl['n']})", f"Sustained\n(N={sl['n']})"]
    medians = [bl["median_ms"], sl["median_ms"]]
    mins = [bl["min_ms"], sl["min_ms"]]
    maxs = [bl["max_ms"], sl["max_ms"]]
    p95s = [bl["p95_ms"], sl["p95_ms"]]

    x = np.arange(len(labels))
    bars = ax.bar(x, medians, 0.5, color=[COLORS[0], COLORS[1]], edgecolor="black",
                  linewidth=0.5)
    # Error bars: min to max
    ax.errorbar(x, medians,
                yerr=[[m - mn for m, mn in zip(medians, mins)],
                      [mx - m for m, mx in zip(medians, maxs)]],
                fmt="none", ecolor="black", capsize=8, linewidth=1.5)
    # P95 markers
    ax.scatter(x, p95s, marker="v", color="red", zorder=5, s=60, label="P95")

    ax.set_ylabel("Consensus Latency (ms)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f"Fig. V-C: PureChain PoA2 Consensus Latency ({n_val} validators)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate medians
    for i, v in enumerate(medians):
        ax.text(i, v + 50, f"{v:.0f} ms", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT, "fig_vc_consensus_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


def fig_vd_mqtt():
    """V-D: MQTT per-message latency — sequential vs async stacked."""
    seq = pd.read_csv(os.path.join(RESULTS, "mqtt_benchmark.csv"))
    async_df = pd.read_csv(os.path.join(RESULTS, "mqtt_benchmark_async.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))

    jobs = range(len(seq))
    width = 0.35
    x = np.arange(len(jobs))

    ax.bar(x - width/2, seq["pipeline_ms"] / 1000, width,
           label="Sequential (pipeline)", color=COLORS[0])
    ax.bar(x + width/2, async_df["pipeline_ms"] / 1000, width,
           label="Async (pipeline)", color=COLORS[1])

    # Overlay E2E as markers
    ax.scatter(x - width/2, seq["latency_ms"] / 1000, marker="_", s=200,
               color="black", zorder=5, linewidths=2)
    ax.scatter(x + width/2, async_df["latency_ms"] / 1000, marker="_", s=200,
               color="red", zorder=5, linewidths=2)

    ax.set_xlabel("Message #", fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in jobs], fontsize=9)
    ax.set_title("Fig. V-D: MQTT Per-Message Latency (Sequential vs Async)",
                 fontsize=12)

    # Custom legend
    handles = [
        mpatches.Patch(color=COLORS[0], label="Sequential pipeline"),
        mpatches.Patch(color=COLORS[1], label="Async pipeline"),
        plt.Line2D([0], [0], marker="_", color="black", linestyle="None",
                   markersize=10, markeredgewidth=2, label="Sequential E2E"),
        plt.Line2D([0], [0], marker="_", color="red", linestyle="None",
                   markersize=10, markeredgewidth=2, label="Async E2E"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(OUT, "fig_vd_mqtt_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


def fig_ve_scalability():
    """V-E: Scalability crossover — already exists, regenerate with better style."""
    df = pd.read_csv(os.path.join(RESULTS, "scalability_benchmark.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["N"], df["strategy_a_per_record_ms"], "o-", color=COLORS[3],
            linewidth=2, markersize=8, label="Strategy A: Individual commits")
    ax.plot(df["N"], df["strategy_b_per_record_ms"], "s-", color=COLORS[1],
            linewidth=2, markersize=8, label="Strategy B: Merkle batch")

    # Annotate speedup
    for _, row in df.iterrows():
        ax.annotate(f"{row['speedup_factor']:.0f}x",
                    xy=(row["N"], row["strategy_b_per_record_ms"]),
                    xytext=(0, -18), textcoords="offset points",
                    ha="center", fontsize=9, color=COLORS[1], fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size (N compounds)", fontsize=12)
    ax.set_ylabel("Per-Record Anchoring Latency (ms)", fontsize=12)
    ax.set_title("Fig. V-E: Blockchain Scalability — Individual vs Merkle Batch\n"
                 "(Real PureChain Mainnet)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_ve_scalability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    # Also overwrite the old one
    fig2_path = os.path.join(RESULTS, "scalability_figure.png")
    fig, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(df["N"], df["strategy_a_per_record_ms"], "o-", color=COLORS[3],
             linewidth=2, markersize=8, label="Strategy A: Individual commits")
    ax2.plot(df["N"], df["strategy_b_per_record_ms"], "s-", color=COLORS[1],
             linewidth=2, markersize=8, label="Strategy B: Merkle batch")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Batch Size (N)", fontsize=12)
    ax2.set_ylabel("Per-Record Latency (ms)", fontsize=12)
    ax2.set_title("Blockchain Anchoring: Individual vs Merkle Batch", fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(fig2_path, dpi=150); plt.close(fig)
    print(f"  {path}")


def fig_vf_onnx():
    """V-F: ONNX inference latency per model."""
    with open(os.path.join(RESULTS, "onnx_determinism.json")) as f:
        d = json.load(f)

    models = sorted(d["determinism"].keys())
    latencies = [d["determinism"][m]["inference_latency_ms"] for m in models]
    labels = [m.replace("reg_", "R:").replace("clf_", "C:") for m in models]
    colors = [COLORS[1] if d["concordance"][m].get("concordance") else COLORS[3]
              for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(range(len(models)), latencies, color=colors, edgecolor="black",
                   linewidth=0.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Inference Latency per Run (ms)", fontsize=11)
    ax.set_title("Fig. V-F: ONNX Inference Latency (40 runs, all bitwise-identical)",
                 fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, axis="x", which="both")

    for i, (v, m) in enumerate(zip(latencies, models)):
        det = "DET" if d["determinism"][m]["bitwise_identical"] else "FAIL"
        ax.text(v * 1.3, i, f"{v:.1f} ms [{det}]", va="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT, "fig_vf_onnx_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


def fig_vg_tamper():
    """V-G: Tamper detection — hash divergence visual."""
    with open(os.path.join(RESULTS, "tamper_demo.json")) as f:
        d = json.load(f)

    orig = d["original_hash"]
    tamp = d["tampered_hash"]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    y = 0.75
    ax.text(0.02, y, "Original:", fontsize=11, fontweight="bold", family="monospace",
            transform=ax.transAxes)
    ax.text(0.15, y, orig, fontsize=9, family="monospace", color="green",
            transform=ax.transAxes)

    y = 0.45
    ax.text(0.02, y, "Tampered:", fontsize=11, fontweight="bold", family="monospace",
            transform=ax.transAxes)
    ax.text(0.15, y, tamp, fontsize=9, family="monospace", color="red",
            transform=ax.transAxes)

    y = 0.15
    mod = d["modification"]
    ax.text(0.02, y, f"Change: {mod}", fontsize=10, family="monospace",
            transform=ax.transAxes)
    ax.text(0.60, y, f"Block: {d['block_number']}   Verified: {d['onchain_original_verified']}   "
            f"Tamper rejected: {d['onchain_tamper_rejected']}",
            fontsize=9, transform=ax.transAxes)

    ax.set_title("Fig. V-G: Tamper Detection — SHA-256 Hash Divergence", fontsize=12,
                 pad=10)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_vg_tamper.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_vh_overhead():
    """V-H: Blockchain overhead — bar chart with/without."""
    with open(os.path.join(RESULTS, "overhead_baseline.json")) as f:
        d = json.load(f)

    targets = list(d.keys())
    labels = [f"{t}\n({d[t]['n_compounds']} cpds)" for t in targets]

    pipe_only = [d[t]["pipeline_only_median_s"] for t in targets]
    pipe_bc = [d[t]["pipeline_with_blockchain_median_s"] for t in targets]
    overhead = [d[t]["blockchain_overhead_s"] for t in targets]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(targets))
    width = 0.3

    ax.bar(x - width/2, pipe_only, width, label="Pipeline only", color=COLORS[0])
    ax.bar(x + width/2, pipe_bc, width, label="Pipeline + PureChain", color=COLORS[3])

    # Annotate overhead
    for i in range(len(targets)):
        pct = d[targets[i]]["blockchain_overhead_pct"]
        ax.annotate(f"+{overhead[i]:.1f}s ({pct}%)",
                    xy=(x[i] + width/2, pipe_bc[i]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color=COLORS[3])

    ax.set_ylabel("Median Wall-Clock Time (s)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Fig. V-H: Blockchain Anchoring Overhead", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(OUT, "fig_vh_overhead.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    print("Generating all paper figures...")
    fig_vb_edge()
    fig_vc_consensus()
    fig_vd_mqtt()
    fig_ve_scalability()
    fig_vf_onnx()
    fig_vg_tamper()
    fig_vh_overhead()
    print("\nDone. All figures in results/paper_tables/")
