## Table V-B: Edge Deployment Performance

| Metric                    |   server |   rpi4 |   jetson_nano |   constrained |
|:--------------------------|---------:|-------:|--------------:|--------------:|
| Latency @ N=1000 (s)      |     10.3 |   10.2 |          11.1 |           9.6 |
| Peak memory (MB)          |    337.6 |  337.9 |         337.2 |         336.9 |
| Throughput (cpm)          |   6595.7 | 6620.9 |        6072   |        7053   |
| Blockchain commit (ms)    |   1820   | 2298.3 |        2744   |        1601.8 |
| Pipeline success rate (%) |    100   |  100   |         100   |         100   |

*All 12 runs executed inside resource-capped `pureprot-edge:latest` containers with real PureChain mainnet commits (block numbers in the CSV). Container CPU enforcement was verified independently with a 4-process Python burn test: cpuset=1 core ran 3× slower than cpuset=4 cores, confirming the constraints are honoured. The narrow throughput spread across tiers reflects the workload mix: per-batch latency is dominated by the ~2 s PureChain commit and the one-time ~1.8 s joblib model load, both I/O-bound and indifferent to core count. The workload includes real RDKit Morgan fingerprint + descriptor computation on CHEMBL243 (HIV-1 protease) study compounds (~4.75 ms/compound, single-threaded). Core-count differentiation becomes prominent with parallelised featurisation (multiprocessing pool). Peak memory of ~337 MB fits under all four tier RAM ceilings.*