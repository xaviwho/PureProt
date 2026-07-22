# A3 - Real Provenance Baselines vs PoA-squared (Reviewer R2.5)

Reviewer 2 (R2.5) flagged the original blockchain baseline (individual commits vs Merkle batch) as a strawman. This experiment compares PureChain PoA-squared anchoring against two *legitimate* provenance mechanisms -- an Ed25519 signed append-only log and IPFS-style content addressing -- plus Merkle batching, across both cost and trust axes.

Local mechanisms measured on `Windows-11-10.0.26200-SP0`, Python 3.12.10, cryptography 49.0.0, best-of-5 single-thread. PoA-squared figures are **reused** from IOT_EXPERIMENT_RESULTS.md V-C (consensus) and V-E (scalability) (no mainnet contact in this run).

## Write latency (per record)

| Mechanism | Write latency / record | Basis |
|---|---|---|
| Ed25519 signed log | 52.3 us | measured, N=100 |
| IPFS CID (content address, local) | 7.4 us | measured, N=100 |
| Merkle root build | 1.1 us | measured, N=100 |
| PoA-squared individual commit | 1,904 ms | reused (V-C) |
| PoA-squared Merkle-batch (amortised) | 21.2 ms | reused (V-E, N=100) |

At N=100, the Ed25519 log is ~**36,371x** faster per record than an individual PoA-squared commit, and local content addressing ~**257,297x** faster. **PoA-squared does not win on latency and we do not claim it does.**

## Trust & property matrix

| Property | Ed25519 signed log | IPFS content addressing | Merkle batch | PoA-squared (PureChain) |
|---|---|---|---|---|
| Trust model | Single key holder (centralised) | Publisher chooses authoritative CID | Inherits its anchor's trust | Multi-validator consortium |
| Fault tolerance | No | No | No (it is a data structure) | Tolerates minority validator faults; majority-honest authority set (NOT classical 1/3-Byzantine BFT) -- to be empirically tested in A4 |
| Rewrite resistance | Key holder can re-sign history | None on its own (no ordering) | Needs an external anchor | Requires majority-of-validators collusion (no single party) |
| Ordering + timestamp | Yes (chained) | No (content-addressed only) | No (unordered set) | Yes (block height + time) |
| Independent public verification | Needs trusted public key | Needs trusted CID index | Needs the anchor | Public tx/state read via RPC (validator-set query disabled on current public endpoint) |
| Tamper-evidence granularity | Per entry | Per object | Per leaf (log2 N proof) | Per committed hash |
| Network / infra dependency | None (local) | IPFS network for persistence* | None | Live validator network |
| Single point of compromise | The private key | The CID publisher | The anchor | None (distributed) |

*IPFS persistence/retrieval was **not** measured here; only the local content-addressing property (CIDv1 raw, sha2-256) was computed. See the honesty note in blockchain/provenance_baselines.py.

## Honest interpretation

- **Where PoA-squared loses:** latency and cost. A signed local log or content addressing is 4-5 orders of magnitude cheaper per record. If the only requirement were tamper-evidence under a trusted operator, an Ed25519 append-only log would be the correct, simpler choice.

- **Where PoA-squared wins (the actual contribution):** it removes the trusted single party. The Ed25519 log's immutability collapses if the key holder is compromised or dishonest; IPFS content addressing gives no ordering, timestamp, or protection against a publisher swapping which CID is 'the' record; Merkle batching still needs *something* to anchor the root -- and that anchor is exactly what PoA-squared provides. PoA-squared delivers multi-validator, majority-honest, publicly verifiable ordering with no single point of key compromise (its fault-tolerance envelope is quantified empirically in A4, not assumed here).

- **Consequence for the paper (R2.5):** the correct framing is not 'PoA-squared is faster' but 'PoA-squared is the right choice *only* when the trust model forbids a single authoritative key holder and requires independent public verifiability; otherwise a signed log is cheaper.' The Merkle-batch result should be presented as an optimisation *within* the PoA-squared anchor, not as a comparison baseline.
