## Table V-C.1: PoA2 Consensus Latency (PureChain mainnet)

| Measurement        |   Median (ms) |   P95 (ms) |   Min (ms) |   Max (ms) |
|:-------------------|--------------:|-----------:|-----------:|-----------:|
| Baseline (20 txs)  |        1970.3 |     2315   |      794.2 |     2843.2 |
| Sustained (50 txs) |        1973.6 |     2233.2 |      813.3 |     2355.9 |

## Table V-C.2: Hash Integrity Verification

| Quantity                       | Value              |
|:-------------------------------|:-------------------|
| Hashes committed & verified    | 20/20 (100%)       |
| Long-term re-verify (first tx) | PASS               |
| Blocks elapsed during test     | 134                |
| Block range                    | 1011678 -> 1011812 |

*All 70 transactions committed to real PureChain mainnet (chain ID 900520900520, RPC: https://purechainnode.com). Baseline and sustained measurements confirm no latency degradation under back-to-back load. Hash integrity is verified by re-reading each committed transaction from the chain and comparing the on-chain resultHash to the locally computed digest.*
