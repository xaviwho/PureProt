# Suggested LaTeX captions for the resubmission figures/tables

IEEE figures are captioned in the LaTeX (`\caption{}`), not on the canvas. These
carry the prose deliberately kept off the figures. Numbers here trace to the
same `results/` files the figure scripts read.

---

**Fig. 1 — Cross-architecture determinism.**
> Byte-level agreement of ONNX inference across three heterogeneous devices
> (x86\_64 server, Jetson Orin Nano, Raspberry Pi 4) at ONNX Runtime 1.18.0 and
> intra-op thread counts $\{1,2,4\}$, versus the x86 reference. 57 of 63 cells are
> byte-identical; the sole divergence is the RBF-kernel SVR regressor on both
> aarch64 devices. **The identical MATCH/DIFFER pattern was observed at every one
> of the five ONNX Runtime versions tested (1.17.3, 1.18.0, 1.19.2, 1.20.1,
> 1.22.0)** — 1.18.0 is shown for clarity (full matrix: `results/determinism/cross_arch.md`).

**Fig. 2 — Real-device edge throughput.**
> Screening throughput (RDKit featurization + ONNX consensus inference + SHA-256)
> on real hardware. Each bar is the **best of 3 runs** per batch size (per-cell
> variation < 3%). The x86 tier is an **uncapped x86\_64 Linux container on
> server-class hardware — a real-hardware reference, not a Docker-resource-capped
> edge simulation.** Jetson Orin Nano at the 25 W power mode; both edge devices
> run from SD-card storage. Companion values (cores, vs.-server ratio, peak RSS,
> inference share) in Table~II.

**Fig. 3 — Provenance write cost.**
> Per-record write latency (log scale) for each provenance mechanism, from
> `results/provenance_comparison.csv` (local mechanisms) and
> `results/scalability_benchmark.csv` (on-chain PoA$^2$, $N{=}100$). This is a
> **cost** comparison only; the trust properties that motivate PoA$^2$ despite its
> higher cost are in Table~I. An individual PoA$^2$ commit costs
> $\sim$35{,}900$\times$ an Ed25519 log append and $\sim$254{,}000$\times$ an IPFS
> content-address (ratios computed from the loaded values).

**Fig. 4 — Validator failure injection and gap-condition recovery.**
> (a) Static 4-signer Clique testnet: block production continues under one
> validator loss (4/4$\to$3/4) but stalls on loss of majority (3/4$\to$2/4), and
> recovers in **1.04 s** once a validator restarts — i.e.\ majority-honest,
> minority-fault-tolerant, not $\tfrac{1}{3}$-Byzantine. (b) The external
> gap-condition controller detects the failed signer's reliability collapse
> $R(\text{v2})\to 0$ **18.6 s** after the fault, promotes a standby signer (first
> standby-sealed block **33.1 s** after the fault), and the healed set then
> survives a second validator loss **without stalling** — the failure that stalled
> the static set in (a). Times/latencies from `results/a4_testnet/*.json`.

---

**Table I — Provenance trust/property matrix (accompanies Fig. 3).**
> Qualitative trust properties per mechanism, with the measured per-record write
> latency for context. PoA$^2$'s advantage is confined to the trust axis (no single
> point of compromise; rewrite needs majority-of-validators collusion); it is the
> most expensive mechanism by far (Fig. 3). "Public verification" for PoA$^2$ is
> public read of transactions/state via RPC; note the Clique validator-set query
> is disabled on the current public endpoint, so signer-set introspection uses
> ecrecover over block seals.

**Table II — Determinism detail (accompanies Fig. 1).**
> Per-model determinism: within-configuration reproducibility (40 runs),
> cross-thread stability $\{1,2,4\}$, cross-ONNX-Runtime-version stability
> (1.17--1.22), and float32/float64 residual vs.\ scikit-learn. Tree-ensemble
> regressors are reproducible within a fixed thread count but differ across thread
> counts; the SVR has no float64 ONNX Runtime kernel; the StandardScaler is exact
> in float64.
