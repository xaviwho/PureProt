# A1 — ONNX Determinism Matrix (R2.2, R2.4)

Cross-ORT-version × thread × 40-run bitwise-hash agreement on the exported models. Same harness runs on aarch64 (Jetson) in B1.

## Environments

| ORT | status | numpy | python | platform |
|---|---|---|---|---|
| 1.16.3 | **install_failed** | — | — | ERROR: Could not find a version that satisfies the requireme |
| 1.17.3 | ok | 1.26.4 | 3.12.10 | AMD64/Windows |
| 1.18.0 | ok | 1.26.4 | 3.12.10 | AMD64/Windows |
| 1.19.2 | ok | 2.5.1 | 3.12.10 | AMD64/Windows |
| 1.20.1 | ok | 2.5.1 | 3.12.10 | AMD64/Windows |
| 1.22.0 | ok | 2.5.1 | 3.12.10 | AMD64/Windows |

## Within-config determinism & cross-thread agreement

Each cell: are all 40 runs identical (within-config), and does the hash stay equal across threads {1,2,4} (cross-thread)?

| Model | 1.17.3 | 1.18.0 | 1.19.2 | 1.20.1 | 1.22.0 |
|---|---|---|---|---|---|
| clf_gb_clf | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable |
| clf_rf_clf | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable |
| clf_svc | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable |
| reg_gradient_boosting | det/THR-VARY | det/THR-VARY | det/THR-VARY | det/THR-VARY | det/THR-VARY |
| reg_random_forest | det/THR-VARY | det/THR-VARY | det/THR-VARY | det/THR-VARY | det/THR-VARY |
| reg_svr | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable |
| scaler | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable | det/thr-stable |

## Cross-ORT-version agreement (threads=1)

Distinct canonical hashes for each model across the successful ORT versions. **1 = byte-identical across every ORT version.**

| Model | distinct hashes across ORT | verdict |
|---|---|---|
| clf_gb_clf | 1 | stable across ORT |
| clf_rf_clf | 1 | stable across ORT |
| clf_svc | 1 | stable across ORT |
| reg_gradient_boosting | 1 | stable across ORT |
| reg_random_forest | 1 | stable across ORT |
| reg_svr | 1 | stable across ORT |
| scaler | 1 | stable across ORT |

_Note: numpy co-varies with ORT where old ORT pins numpy<2; see Environments table. Hash drift may reflect ORT and/or numpy changes._