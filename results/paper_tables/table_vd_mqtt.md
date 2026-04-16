## Table V-D.1: MQTT End-to-End Latency Breakdown

| Stage                                         |   Median (ms) |   P95 (ms) |
|:----------------------------------------------|--------------:|-----------:|
| Per-message pipeline (inference + blockchain) |          2009 |       7121 |
| End-to-end including queue wait               |         19706 |      27747 |

## Table V-D.2: Throughput Summary

- **Messages measured:** 10
- **Message rate:** 0.05 msg/s (2.5 compounds/s)
- **Blockchain commit success:** 100% (10/10)

*All 10 messages processed via real Eclipse Mosquitto 2.0 broker (Docker container) with PureChain mainnet blockchain commits. Blockchain commit success: 100%. Block range: 1011650--1011668. Pipeline inference includes full sklearn consensus prediction over 50 compounds per message.*
