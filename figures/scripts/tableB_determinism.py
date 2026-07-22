#!/usr/bin/env python3
"""TABLE B — Determinism detail (R2.4).

All derived from results/determinism/determinism_matrix.csv (per-model x ORT x
thread canonical hashes + bitwise_identical) and dtype_concordance.json.
Columns: within-config (40 runs), cross-thread, cross-ORT (1.17-1.22), dtype residual.
"""
import csv, json, os, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DET = os.path.join(ROOT, "results", "determinism")
PRETTY = {"reg_svr": "SVR (reg)", "reg_random_forest": "RandomForest (reg)",
          "reg_gradient_boosting": "GradBoosting (reg)", "clf_svc": "SVC (clf)",
          "clf_rf_clf": "RandomForest (clf)", "clf_gb_clf": "GradBoosting (clf)",
          "scaler": "StandardScaler"}
ORDER = ["reg_svr", "reg_random_forest", "reg_gradient_boosting",
         "clf_svc", "clf_rf_clf", "clf_gb_clf", "scaler"]


def main():
    rows = list(csv.DictReader(open(os.path.join(DET, "determinism_matrix.csv"))))
    dtype = json.load(open(os.path.join(DET, "dtype_concordance.json")))["results"]
    print("TABLE B sources: determinism_matrix.csv, dtype_concordance.json")

    within = {}          # all 40-run cells bitwise identical?
    by_mt = defaultdict(set)   # (model,ort)->set(hash) across threads
    by_m1 = defaultdict(set)   # model->set(hash) at threads=1 across ORT
    for r in rows:
        m = r["model"]
        within.setdefault(m, True)
        if r["bitwise_identical"].lower() != "true":
            within[m] = False
        by_mt[(m, r["ort_version"])].add(r["canonical_hash"])
        if int(r["threads"]) == 1:
            by_m1[m].add(r["canonical_hash"])

    def dtype_resid(m):
        d = dtype.get(m, {})
        f32 = d.get("float32", {}); f64 = d.get("float64", {})
        parts = []
        if f32.get("exported"):
            parts.append(f"f32 {f32['max_abs_diff_vs_sklearn']:.2e}")
        if f64.get("exported"):
            v = f64["max_abs_diff_vs_sklearn"]
            parts.append("f64 exact (0)" if v == 0 else f"f64 {v:.2e}")
        elif "float64" in d and not f64.get("exported"):
            parts.append("f64 n/a$^\\dagger$")
        return "; ".join(parts) if parts else "--"

    models = [m for m in ORDER if m in within]
    tex = [r"\begin{tabular}{lllll}", r"\toprule",
           r"Model & Within-config & Cross-thread & Cross-ORT & dtype residual \\",
           r" & (40 runs) & \{1,2,4\} & (1.17--1.22) & (vs sklearn) \\", r"\midrule"]
    md = ["| Model | Within-config (40 runs) | Cross-thread {1,2,4} | Cross-ORT (1.17-1.22) | dtype residual vs sklearn |",
          "|---|---|---|---|---|"]
    for m in models:
        wc = "identical" if within[m] else "NON-det"
        ct = "stable" if all(len(by_mt[(m, o)]) == 1 for o in {r["ort_version"] for r in rows}) else "varies"
        co = "stable" if len(by_m1[m]) == 1 else f"{len(by_m1[m])} variants"
        dr = dtype_resid(m)
        print(f"  {m}: within={wc} cross-thread={ct} cross-ORT={co} dtype='{dr}'")
        tex.append(f"{PRETTY[m]} & {wc} & {ct} & {co} & {dr} \\\\")
        md.append(f"| {PRETTY[m]} | {wc} | {ct} | {co} | {dr.replace('$^\\\\dagger$','†').replace('$','')} |")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"% $^\dagger$ ONNX Runtime has no float64 SVMRegressor kernel (NOT_IMPLEMENTED);",
            r"% tree float64/float32 re-export failed under the pinned onnx 1.17 toolchain."]
    open(os.path.join(ROOT, "figures", "tables", "tableB_determinism.tex"), "w").write("\n".join(tex))
    open(os.path.join(ROOT, "figures", "tables", "tableB_determinism.md"), "w").write("\n".join(md))
    print("TABLE B output: figures/tables/tableB_determinism.{tex,md}")


if __name__ == "__main__":
    main()
