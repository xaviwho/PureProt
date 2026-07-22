#!/usr/bin/env python3
"""TABLE A — Provenance trust/property matrix (R2.5).

Qualitative trust cells are PARSED from results/provenance_comparison.md (the
designed comparison; there is no numeric raw file for 'who can rewrite history').
The write-latency column is DERIVED from the CSVs (provenance_comparison.csv +
scalability_benchmark.csv) so the cost loss is quantified from data, not typed.
Emits LaTeX (booktabs) + a markdown preview.
"""
import csv, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
MD = os.path.join(RES, "provenance_comparison.md")

# brief-specified property subset (md row label -> short column header)
PROPS = [("Single point of compromise", "Single point of compromise"),
         ("Rewrite resistance", "Rewrite resistance"),
         ("Ordering + timestamp", "Ordering \\& timestamp"),
         ("Independent public verification", "Public verification")]
# md column order: Ed25519 | IPFS | Merkle | PoA2
MECHS = ["Ed25519 signed log", "IPFS content addressing", "Merkle batch", "PoA$^2$ (PureChain)"]


def parse_md_trust():
    rows = {}
    for line in open(MD, encoding="utf-8"):
        if line.startswith("| ") and " | " in line and not line.startswith("| Property") and "---" not in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5:
                rows[cells[0]] = cells[1:]   # property -> [ed25519, ipfs, merkle, poa2]
    return rows


def latency_col():
    def row_at(p, n):
        for r in csv.DictReader(open(p)):
            if int(r["N"]) == n: return r
    prov = row_at(os.path.join(RES, "provenance_comparison.csv"), 100)
    scal = row_at(os.path.join(RES, "scalability_benchmark.csv"), 100)
    return {
        "Ed25519 signed log": f"{float(prov['ed25519_write_ms_per_rec']):.4f}",
        "IPFS content addressing": f"{float(prov['cid_write_ms_per_rec']):.4f}",
        "Merkle batch": f"{float(prov['merkle_root_ms_per_rec']):.4f}",
        "PoA$^2$ (PureChain)": f"{float(scal['strategy_a_per_record_ms']):,.0f} (indiv.) / "
                               f"{float(scal['strategy_b_per_record_ms']):.0f} (batch)",
    }


def clean(s):
    s = re.sub(r"\*+", "", s)
    return s.replace("PoA-squared", "PoA$^2$").replace("--", "--")


def main():
    trust = parse_md_trust()
    lat = latency_col()
    missing = [p for p, _ in PROPS if p not in trust]
    if missing:
        raise SystemExit(f"STOP: trust rows missing from md: {missing}")
    print("TABLE A source: results/provenance_comparison.md (trust rows) +",
          "provenance_comparison.csv/scalability_benchmark.csv (latency)")

    # transpose: rows = mechanisms, cols = properties + latency
    def cell(prop, mi): return clean(trust[prop][mi])
    latency_hdr = "Write latency (ms/rec)"
    headers = [h for _, h in PROPS] + [latency_hdr]

    # LaTeX (double-column, wrapped p{} columns)
    colspec = "p{1.7cm}" + "p{2.4cm}" * len(PROPS) + "p{2.2cm}"
    tex = [r"\begin{tabular}{" + colspec + "}", r"\toprule",
           "Mechanism & " + " & ".join(headers) + r" \\", r"\midrule"]
    for mi, mech in enumerate(MECHS):
        cells = [cell(p, mi) for p, _ in PROPS] + [lat[mech]]
        tex.append(mech + " & " + " & ".join(cells) + r" \\")
        tex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}"]
    p_tex = os.path.join(ROOT, "figures", "tables", "tableA_trust_matrix.tex")
    open(p_tex, "w", encoding="utf-8").write("\n".join(tex))

    # markdown preview
    md = ["| Mechanism | " + " | ".join(headers) + " |",
          "|" + "---|" * (len(headers) + 1)]
    for mi, mech in enumerate(MECHS):
        cells = [cell(p, mi) for p, _ in PROPS] + [lat[mech]]
        md.append("| " + mech.replace("$^2$", "²") + " | " + " | ".join(cells) + " |")
    p_md = os.path.join(ROOT, "figures", "tables", "tableA_trust_matrix.md")
    open(p_md, "w", encoding="utf-8").write("\n".join(md))

    print("TABLE A rows:")
    for mi, mech in enumerate(MECHS):
        print(f"  {mech}: SPoC='{cell('Single point of compromise', mi)}' | latency={lat[mech]} ms")
    print("TABLE A output:", p_tex, "+", p_md)


if __name__ == "__main__":
    main()
