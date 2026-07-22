# PureProtX — IEEE IoT-J Resubmission: Master Report & Implementation Guide

**Manuscript:** IoT-65232-2026 ("Reject with encouragement to resubmit")
**Compiled:** 2026-07-21 · **Single source of truth** — supersedes the individual
review-response MDs.

> **How to use this document.** §2 is the reviewer→evidence→action map. §3 is the
> condensed evidence for each experiment (with the raw-data file paths). §4–§6 are
> the concrete manuscript/code changes to implement. §7–§10 cover environment,
> on-chain proof, artifacts, and honesty caveats. §11 lists what still needs a
> decision from Xavi. Wording is **proposed** — Xavi accepts/edits the manuscript.

Legend: ✅ evidence produced · ✍️ manuscript edit · 🔧 code/repo fix · ❓ decision needed

---

## 1. Context

- **Contribution reframing:** the core claim is **independently verifiable
  deterministic re-execution** — canonical serialization + on-chain pinning of the
  model hash *and* producing configuration, so any third party can re-run and
  cryptographically confirm byte-identical results. Drug-discovery virtual
  screening is the **benchmark/demonstrator**, not the contribution.
- **Ground truth found during this work:** PureChain runs **stock go-ethereum
  v1.13.15 Clique** (not a novel consensus); its "PoA²/gap condition" is a
  policy/layer, not node code. This is disclosed honestly and turned into a
  strength (deployable on commodity chains).
- **Devices used:** x86_64 dev box (Windows), **Jetson Orin Nano** (aarch64,
  JetPack 6.2), **Raspberry Pi 4 Model B** (aarch64, Debian 13). Both edge devices
  on SD card (no NVMe available).

---

## 2. Reviewer map — comment → evidence → action

| # | Reviewer point | Evidence (§3) | Manuscript action (§4) |
|---|---|---|---|
| **R1.1** | Scalability > 10⁶ untested | — | ✍️ keep as future work |
| **R1.2** | Live validator failure injection missing | ✅ §3.5 real Clique testnet | ✍️ replace with measured envelope + gap-condition demo |
| **R1.3** | Generalization to other domains | ✅ §3.6 | ✍️ add generalization section |
| **R2.1** | Novelty unclear (combination of tools) | ✅ §3.5 + §3.6 | ✍️ reframe as property; disclose stock Clique |
| **R2.2** | Determinism across heterogeneous nodes | ✅ §3.1 + §3.2 (3 devices) | ✍️ rescope claim (§4.1) |
| **R2.3** | Edge used Docker caps, not real devices | ✅ §3.3 (Jetson + Pi4, on-chain) | ✍️ replace Table II / Fig 4 |
| **R2.4** | ONNX bitwise determinism overstated | ✅ §3.1 + §3.2 | ✍️ delete false claim; scope it (§4.1) |
| **R2.5** | Blockchain baseline is a strawman | ✅ §3.4 real provenance baselines | ✍️ reframe trust-vs-cost (§4.3) |

**Bottom line:** every reviewer comment now has real evidence except R1.1 (future
work by design). Two writing tasks (A2 rescope, A5 novelty) are drafted in §4.

---

## 3. Experiments & findings

Each subsection lists the **finding**, the **raw-data files**, and the **scripts**.

### 3.1 A1 — ONNX determinism characterization (R2.2, R2.4)  ✅

Harness ran each exported ONNX model 40× under a matrix of **ORT version ×
intra-op threads × dtype**, vs an sklearn reference.

- **Within a pinned thread count: bitwise-deterministic** — 40/40 identical, all 7 models, every ORT version.
- **Byte-identical across ORT 1.17.3→1.22.0** *and* across NumPy 1.26↔2.5 (at fixed threads) — stronger than the paper claimed.
- **Thread count breaks it for tree ensembles:** `reg_random_forest` and `reg_gradient_boosting` differ at 1 vs 2 vs 4 threads (every ORT version). SVR, scaler, all 3 classifiers are thread-stable.
- **dtype:** SVR float32-vs-sklearn gap = **1.25×10⁻⁴** and is **irreducible** (ORT has no float64 `SVMRegressor` kernel — `NOT_IMPLEMENTED`); scaler float64 = **exact (0.0)**.
- **1.16.3** has no cp312 wheel (recorded, not worked around).

Data: `results/determinism/determinism_matrix.{md,csv}`, `harness_ort*.json`,
`dtype_concordance.json`. Scripts: `experiments/determinism_harness.py`,
`run_determinism_matrix.py`, `determinism_dtype.py`.

### 3.2 B1 — cross-architecture determinism, 3 devices (R2.2/R2.3/R2.4)  ✅

Same harness on aarch64, numpy pinned per version to match x86 (architecture is
the only variable).

- **x86 vs Jetson**, 5 ORT versions × 7 models × 3 threads = **105 cells: 90 MATCH, 15 DIFFER** — the 15 DIFFER are *exactly* `reg_svr`.
- **Second aarch64 device (Raspberry Pi 4):** byte-identical to the Jetson for **all 21 cells (SVR included)**.
- **Conclusion:** across **three heterogeneous devices**, 20/21 models are byte-identical; the sole exception (RBF-kernel SVR) is **architecture-level, not device-level** (x86 `b7d5c5be…` vs both aarch64 devices `d01eff08…`). Direct evidence for "heterogeneous edge environments."

Data: `results/determinism/cross_arch.md`, `harness_ort*_arm.json`,
`harness_ort1.18.0_pi4.json`. Scripts: `run_arm_determinism_matrix.sh`,
`compare_cross_arch.py`.

### 3.3 B1 — real-device edge throughput (R2.3)  ✅

Full pipeline: RDKit featurize (2058 feats) → ONNX consensus → SHA-256 →
**PureChain mainnet commit**. Batch sizes 100/500/1000, compute-only throughput.

| Tier | Throughput | vs server | Peak RSS | Infer share |
|---|---|---|---|---|
| **x86_64 server** (12 core, uncapped container) | ~**5,250 compounds/min** | 1.0× | ≤ 446 MB | ~82% |
| **Jetson Orin Nano** (6 core) | ~**3,450 cpm** | **1.5× slower** | ≤ 459 MB | ~73% |
| **Raspberry Pi 4** (4× A72) | ~**2,310 cpm** | **2.3× slower** | ≤ 436 MB | ~73% |

- Throughput flat across batch size on all tiers; a clean server→Jetson→Pi4 gradient.
- **No OOM on real hardware** (≤0.46 GB) — corrects the *simulated* "constrained 512 MB tier OOM" (Docker-cap artifact).
- **9 mainnet transactions**, blocks **1708399–1710453**, independently verified (status 1). See §8.
- **x86 "server" tier** is a real x86_64 **Linux container with no resource caps** (the actual server hardware, not a capped edge simulation) — needed because Windows Application Control blocks the RDKit DLL on the bare Windows host. This is categorically different from the capped Docker tiers R2.3 criticised, and should be labelled as such in the paper.

Data: `results/edge/edge_x86.json`, `edge_jetson.json`, `edge_pi4.json`. Scripts:
`experiments/edge_throughput.py`, `edge_device_run.sh`.

### 3.4 A3 — real provenance baselines (R2.5)  ✅

Compared PureChain PoA² against **Ed25519 signed append-only log**, **IPFS content
addressing**, and Merkle batching, on cost **and** trust axes.

- **Cost:** Ed25519 log ~**37,000×** faster per record than a PoA² commit; local content addressing ~**260,000×** faster. **PoA² does not win on latency.**
- **Trust:** PoA² wins only where it must — no single authoritative key holder, majority-honest multi-validator ordering, public verifiability. The signed log collapses to one key; IPFS has no ordering; Merkle needs an anchor (which *is* PoA²).

Data: `results/provenance_comparison.{md,csv}`, `provenance_manifest.json`.
Scripts: `blockchain/provenance_baselines.py`, `experiments/provenance_benchmark.py`.

### 3.5 A4 — validator failure injection + gap condition (R1.2, R2.1)  ✅

Built a real **4-signer geth v1.13.15 Clique** testnet (chain id 900520900520).

- **Fault-tolerance envelope:** 4/4 healthy (0.5 blk/s); kill 1 → **liveness holds** (3/4); kill 2 → **chain stalls** (2/4, majority lost); restart → **recovery 1.04 s**. → *majority-honest, minority-fault-tolerant — NOT classical ⅓-BFT.*
- **Gap-condition controller** (external layer over stock Clique): detects a dead signer (18.6 s), promotes a standby via `clique_propose` (33.1 s), so a second fault that stalled the static set does **not** stall the healed set.
- **Key disclosure:** the gap condition is a **proposed mechanism validated as an external control layer**, not a PureChain node feature (PureChain = stock Clique). Frame accordingly (R2.1).

Data: `results/a4_testnet/failure_injection.json`, `gap_controller.json`.
Scripts: `blockchain/testnet/{make_genesis.py,run_failure_injection.py,run_gap_controller.py}`.

### 3.6 A5 — novelty + generalization (R2.1, R1.3)  ✍️ (writing)

- **Novelty as a property, not a toolchain:** the coupling of canonical
  serialization + on-chain config/model pinning + hash anchoring into
  *independently verifiable deterministic re-execution*. Built from commodity
  parts (a deployability strength); PureChain = stock Clique (stated plainly).
- **Generalization:** applies to any workload with (1) deterministic inference in
  a pinned config, (2) canonically-serializable outputs, (3) a permissioned chain.
  Example domains: environmental IoT, industrial predictive maintenance, medical
  edge (ECG/EEG), agricultural phenotyping. Drug screening = benchmark.

### 3.7 B0 — ARM64 wheel audit (prerequisite / R2.3)  ✅

On the Jetson (no sudo used; standalone CPython where needed):

- **Python:** device shipped 3.10.12 (Jetson) / 3.13.5 (Pi4); paper pins 3.12.10.
  Installed a **no-sudo, SHA256-verified standalone CPython 3.12.10** on both.
- **At Python 3.10, `scikit-learn 1.8` is impossible** (requires ≥3.11) → skl2onnx
  pulled 1.7.2. **At 3.12.10 the entire pinned Python stack installs cleanly** on
  aarch64: paho-mqtt, web3, scikit-learn 1.8.0, skl2onnx 1.17.0, onnxruntime
  1.18.0, rdkit 2025.9.4. onnxruntime and RDKit — the two "expected to fail" — have
  aarch64 wheels.
- **Only genuine dependency gap: AutoDock Vina** (no wheel any platform) — see §6.

Data: `arm64_audit_log.txt` (repo + Jetson `~/pureprotx/`).

---

## 4. Manuscript changes to implement

### 4.1 Determinism claim rewrite (A2 — the key wording)  ✍️
- **Delete:** "all ONNX operators use IEEE-754 deterministic rounding."
- **Replace with:** *"Inference is bitwise-deterministic within a pinned runtime
  configuration — a fixed ONNX Runtime version and intra-op thread count — anchored
  on-chain so any verifier reproduces the exact configuration. Empirically, outputs
  were byte-identical across ONNX Runtime 1.17–1.22, NumPy 1.26→2.5, and three
  heterogeneous devices (x86_64, Jetson Orin Nano aarch64, Raspberry Pi 4 aarch64)
  at a fixed thread count — with two characterized exceptions: tree-ensemble
  regressors differ across intra-op thread counts, and the RBF-kernel SVR differs
  across CPU architecture."*
- **Anchor** intra-op thread count + NumPy major version alongside ORT version + model hash.
- **State** the SVR limitation (no float64 SVR kernel in ORT; float32 kernel is arch-dependent) — scope SVR determinism per-architecture, or exclude it from the byte-level guarantee.

### 4.2 Novelty / generalization / abstract  ✍️
- Reframe the contribution as the **property** in §3.6; state "built from commodity components; PureChain is stock geth Clique."
- Add the **generalization** subsection (3 requirements + domains).
- **Abstract reorder** — lead with verifiable deterministic computation:
  > *"We present PureProtX, a framework for independently verifiable deterministic
  > re-execution of classical-ML inference on IoT/edge nodes. Outputs are
  > canonically serialized, hashed, and anchored on a permissioned blockchain with
  > the exact producing configuration (model hash, runtime version, thread count),
  > so any third party can re-execute and cryptographically confirm byte-identical
  > results. We characterize the determinism envelope across runtime versions,
  > thread counts, and CPU architectures, and benchmark on drug-discovery virtual
  > screening as a demonstrator."*
- Reframe drug discovery as "benchmark/demonstrator" throughout.

### 4.3 Provenance reframe (R2.5)  ✍️
- State the trade-off: *"PoA² is the right choice only when the trust model forbids
  a single authoritative key holder and requires independent public verifiability;
  otherwise a signed log is cheaper."*
- Present Merkle batching as an **optimisation within** the PoA² anchor, not a baseline.

### 4.4 Failure / PoA² framing (R1.2, R2.1)  ✍️
- Replace observational/simulated failure text with the measured envelope (§3.5).
- Present the gap condition as a **proposed mechanism validated as an external
  control layer** over stock Clique — not node consensus.
- **PoA² is majority-honest, not classical ⅓-BFT** — remove any BFT overclaim.

### 4.5 Factual / labeling fixes  ✍️
- **"Jetson Nano" → "Jetson Orin Nano"** throughout (cited device is EOL).
- **Eq. numbering:** gap condition cited as Eq. 5–7 (CLAUDE.md) vs "Eq. 8" (code) — reconcile.
- **V-C reproducibility note:** `clique_getSigners` is disabled on today's public
  RPC; update method to recover signers via ecrecover over block seals.
- **No-OOM correction:** retract/relabel the simulated "constrained 512 MB OOM" (real devices peak ≤ 0.46 GB).

---

## 5. Code & environment fixes

- 🔧 **DONE:** `pureprot/ai_model.py` — `Descriptors.FractionCsp3` → `FractionCSP3`
  (rdkit 2025.9.4 renamed it; featurization was broken under the pin).
- 🔧 **Recommended:** modernize `GetMorganFingerprintAsBitVect` → `MorganGenerator` (deprecation warning only).
- 🔧 **`requirements.txt` is stale** vs the pins (sklearn 1.2.2, web3 6.19.0, numpy
  1.23.5, rdkit-pypi 2022.9.5). A verified **`requirements-repro.txt`** was added
  for the determinism/edge stack; regenerating the legacy file needs a
  full-codebase test run (older modules may need old APIs) — ❓ Xavi to green-light.
- 🔧 **Pin onnxruntime ≥ 1.19.2** (numpy-2 native) to remove the 1.18.0-vs-numpy-2
  ABI conflict; A1 proved output is byte-identical (see §7).

---

## 6. AutoDock Vina on aarch64 — ❓ decision needed

`vina==1.2.7` has **no wheel for any platform**; options for the edge devices:
(1) **conda-forge** linux-aarch64 build (likely exists; needs miniforge on device);
(2) **source build** (Boost + SWIG + compile; needs sudo/toolchain); (3) prebuilt
release binaries are **x86_64/macOS only** (no Linux-aarch64).

**Recommendation — question the need first:** the B1 edge pipeline is **AI
inference**; it never calls Vina. Docking is part of the x86 demonstrator and the
hybrid result was null (Wilcoxon p = 1.000). **If no edge/IoT claim requires
on-device docking, Vina-on-aarch64 is a non-blocker** — keep it x86-only. Only if
docking must run on the Jetson/Pi4 do we take the conda-forge route.

---

## 7. Environment / reproducibility

- **Pinned env:** `requirements-repro.txt` (verified across x86 + Jetson + Pi4, Python 3.12.10).
- **NumPy ABI conflict (important):** onnxruntime 1.18.0 needs numpy<2, but the
  saved joblib models were pickled under numpy 2.x → they cannot coexist in one
  env. A1 shows ONNX output is byte-identical ORT 1.17→1.22, so **pin onnxruntime
  1.19.2 (numpy-2 native)** to resolve it without changing results. Pin
  `scipy==1.18.0` (currently unpinned).
- **No-sudo device Python:** standalone CPython 3.12.10 (Astral python-build-standalone, SHA256-verified) at `~/pureprotx/py312` on both devices.
- **geth:** v1.13.15 (official release c5ba367e) on x86 (`tools/geth/geth.exe`),
  Jetson, and via `ethereum/client-go:v1.13.15` for the A4 testnet. Note PureChain
  mainnet runs a self-built v1.13.15 (commit `df31f81f`) — same release, different build.

---

## 8. On-chain evidence

- **Edge commits (§3.3):** 9 transactions (Jetson + Pi4 + x86-server), blocks
  **1708399–1710453**, contract `0xb8eb…d56C`, wallet `0xdE8a…f895`. Spot-verified
  live (status 1) on all three tiers.
- Verify any tx: `python tools/verify_onchain.py --tx <hash>`.
- **Prior on-chain evidence** (pre-existing, IOT_EXPERIMENT_RESULTS.md): ~350 tx,
  blocks 1010803–1012680 (V-B…V-H). A4 testnet is **local only** (no mainnet
  failure injection).

---

## 9. Artifacts & scripts index

| Path | Purpose |
|---|---|
| `experiments/determinism_harness.py` | ONNX determinism harness (cross-platform) |
| `experiments/run_determinism_matrix.py` · `run_arm_determinism_matrix.sh` | ORT-version matrix (x86 / ARM) |
| `experiments/determinism_dtype.py` · `compare_cross_arch.py` | dtype axis · cross-arch diff |
| `experiments/edge_throughput.py` · `edge_device_run.sh` | real-device edge pipeline |
| `experiments/provenance_benchmark.py` · `blockchain/provenance_baselines.py` | provenance baselines |
| `blockchain/testnet/*.py` | geth Clique testnet + failure/gap-condition |
| `tools/geth/geth.exe` | pinned geth binary (x86) |
| `results/determinism/`, `results/edge/`, `results/a4_testnet/`, `results/*.{md,csv,json}` | raw evidence |
| `requirements-repro.txt`, `arm64_audit_log.txt` | environment / audit log |

---

## 10. Honesty flags / limitations (consolidated)

1. **SVR is the one determinism exception** — varies by thread count *and* CPU
   architecture; its float32 residual is irreducible (no float64 ORT kernel).
2. **x86 server edge baseline** obtained via an **uncapped x86_64 Linux container**
   (Windows Application Control blocks RDKit on the bare host). It is the real
   server hardware, not a capped edge simulation — label it that way to avoid
   conflation with the R2.3-criticised Docker-cap tiers.
3. **Edge devices on SD card** (no NVMe) — recorded; compute-bound so minor effect.
4. **A4 gap condition is an external emulation**, not a PureChain node feature.
5. **PoA² numbers reused** in A3 (no fresh mainnet contact there); A4 testnet is local.
6. **numpy co-varies with ORT** in A1 (ABI) — recorded per cell; byte-identity held regardless.
7. **Repo/config drift found:** `ai_model.py` FractionCsp3 (fixed), stale `requirements.txt`, Morgan API deprecation, Eq numbering, V-C RPC method.

---

## 11. Open decisions for Xavi

1. **Manuscript wording** — accept/edit §4.1–§4.5 (I don't edit the manuscript itself).
2. **`requirements.txt` regen** — green-light a full-codebase test pass to bump legacy pins?
3. **Vina** — confirm whether on-device docking is required (likely non-blocker, §6).
4. **onnxruntime pin** — adopt 1.19.2 (recommended) or keep 1.18.0 + re-pickle models?
5. **Git** — commit this body of work to a branch? (currently all untracked).

---

*This report consolidates and replaces: the five `REVIEW_RESPONSE_*.md`,
`RESUBMISSION_CHECKLIST.md`, `ENV_PINNING_NOTE.md`, and `arm64_wheel_audit.md`
(repo copy). Raw data files under `results/` and the scripts remain as evidence.*
