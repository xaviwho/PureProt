#!/usr/bin/env python3
"""
B1 — Cross-architecture determinism comparison (Reviewer R2.2, R2.3, R2.4).

Given determinism-harness JSON outputs from two or more machines (e.g. x86_64 and
the Jetson aarch64), reports whether each (model, thread-count) cell produced a
byte-identical hash ACROSS architectures. This is the decisive test of the
paper's cross-platform determinism claim.

The verdict is deliberately blunt: for each cell it prints MATCH (bitwise-identical
across arch) or DIFFER (architecture changes the bytes). A DIFFER result is a
legitimate, publishable finding — it bounds the determinism claim to a single
architecture, which the paper must then state.

Usage:
  python experiments/compare_cross_arch.py results/determinism/harness_ort1.18.0.json \
                                            results/determinism/harness_ort1.18.0_arm.json
  # or point it at a directory / glob of harness_*.json files
  python experiments/compare_cross_arch.py "results/determinism/harness_ort1.18.0*.json"
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MD = os.path.join(ROOT, "results", "determinism", "cross_arch.md")


def load(paths):
    envs = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        m = d["manifest"]
        envs.append({
            "path": p,
            "arch": m.get("machine", "?"),
            "ort": m.get("onnxruntime_version", "?"),
            "numpy": m.get("numpy_version", "?"),
            "system": m.get("system", "?"),
            "results": d["results"],
        })
    return envs


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(2)
    # expand globs / dirs
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += sorted(glob.glob(os.path.join(a, "harness_*.json")))
        else:
            paths += sorted(glob.glob(a))
    paths = [p for p in paths if os.path.exists(p)]
    if len(paths) < 2:
        sys.exit(f"need >=2 harness JSONs to compare; got {paths}")

    envs = load(paths)
    archs = sorted({e["arch"] for e in envs})
    lines = ["# B1 — Cross-Architecture Determinism (R2.2, R2.3, R2.4)\n"]
    lines.append("Byte-identical hash agreement per (model, thread-count) across "
                 "architectures. **MATCH = bitwise-identical across arch.**\n")
    lines.append("## Environments compared\n")
    lines.append("| file | arch | system | ORT | numpy |")
    lines.append("|---|---|---|---|---|")
    for e in envs:
        lines.append(f"| {os.path.basename(e['path'])} | {e['arch']} | {e['system']} | {e['ort']} | {e['numpy']} |")
    lines.append("")

    if len(archs) < 2:
        lines.append("> ⚠️ All inputs are the **same** architecture "
                     f"({archs}); cross-arch comparison is not meaningful until an "
                     "aarch64 harness JSON is added.\n")

    # union of models & thread keys
    models = sorted({m for e in envs for m in e["results"]})
    thread_keys = sorted({tk for e in envs for m in e["results"].values() for tk in m})

    # Group envs by ORT version and compare architectures WITHIN each version, so
    # ORT versions are never conflated. A cell MATCHes only if every architecture
    # present for that (version, model, threads) shares one hash.
    versions = sorted({e["ort"] for e in envs})
    n_match = n_differ = 0
    lines.append("## Cross-architecture hash agreement (per ORT version)\n")
    for ver in versions:
        vhosts = [e for e in envs if e["ort"] == ver]
        varchs = sorted({e["arch"] for e in vhosts})
        if len(varchs) < 2:
            lines.append(f"### ORT {ver} — only {varchs} present (no cross-arch pair, skipped)\n")
            continue
        lines.append(f"### ORT {ver}  ({' vs '.join(varchs)})\n")
        lines.append("| Model | threads | " + " | ".join(varchs) + " | verdict |")
        lines.append("|---" * (len(varchs) + 3) + "|")
        for model in models:
            for tk in thread_keys:
                by_arch = {}
                for e in vhosts:
                    c = e["results"].get(model, {}).get(tk)
                    if c:
                        by_arch[e["arch"]] = c["canonical_hash"]
                if len(by_arch) < 2:
                    continue
                cells = [by_arch.get(a, "—")[:12] if by_arch.get(a) else "—" for a in varchs]
                if len(set(by_arch.values())) == 1:
                    verdict = "**MATCH**"; n_match += 1
                else:
                    verdict = "**DIFFER**"; n_differ += 1
                lines.append(f"| {model} | {tk.replace('threads_','')} | " +
                             " | ".join(cells) + f" | {verdict} |")
        lines.append("")

    lines.append(f"## Summary\n\n- cells MATCH across arch: **{n_match}**\n"
                 f"- cells DIFFER across arch: **{n_differ}**\n")
    if n_differ:
        lines.append("> A non-zero DIFFER count means ONNX inference is **not** "
                     "bitwise-identical across CPU architectures for those cells; "
                     "the determinism claim must be scoped per-architecture (the "
                     "producing arch is anchored on-chain alongside ORT+threads).")
    else:
        lines.append("> All compared cells are byte-identical across architectures "
                     "— the strongest possible cross-platform determinism result.")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[cross-arch] {n_match} MATCH, {n_differ} DIFFER across {archs}")
    print(f"[cross-arch] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
