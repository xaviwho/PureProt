## Table V-E: Merkle Batching vs. Individual Commits

|   N |   Strategy A (ms/record) |   Strategy B (ms/record) |   Speedup (x) |
|----:|-------------------------:|-------------------------:|--------------:|
|  10 |                  1720.09 |                  199.276 |           8.6 |
| 100 |                  1990.23 |                   21.19  |          93.9 |

**Crossover point N\* = 10** (first batch size at which Merkle batching is strictly faster per record than individual commits).

Figure: `results/scalability_figure.png`

*Measured against real PureChain mainnet (chain ID 900520900520) with 1 repeat per N. Strategy A latency is dominated by the ~2 s PureChain consensus latency per transaction; Strategy B always submits exactly one transaction regardless of N, so its per-record cost falls inversely with N.*
