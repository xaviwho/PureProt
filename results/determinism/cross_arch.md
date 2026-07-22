# B1 — Cross-Architecture Determinism (R2.2, R2.3, R2.4)

Byte-identical hash agreement per (model, thread-count) across architectures. **MATCH = bitwise-identical across arch.**

## Environments compared

| file | arch | system | ORT | numpy |
|---|---|---|---|---|
| harness_ort1.17.3.json | AMD64 | Windows | 1.17.3 | 1.26.4 |
| harness_ort1.17.3_arm.json | aarch64 | Linux | 1.17.3 | 1.26.4 |
| harness_ort1.18.0.json | AMD64 | Windows | 1.18.0 | 1.26.4 |
| harness_ort1.18.0_arm.json | aarch64 | Linux | 1.18.0 | 1.26.4 |
| harness_ort1.19.2.json | AMD64 | Windows | 1.19.2 | 2.5.1 |
| harness_ort1.19.2_arm.json | aarch64 | Linux | 1.19.2 | 2.5.1 |
| harness_ort1.20.1.json | AMD64 | Windows | 1.20.1 | 2.5.1 |
| harness_ort1.20.1_arm.json | aarch64 | Linux | 1.20.1 | 2.5.1 |
| harness_ort1.22.0.json | AMD64 | Windows | 1.22.0 | 2.5.1 |
| harness_ort1.22.0_arm.json | aarch64 | Linux | 1.22.0 | 2.5.1 |

## Cross-architecture hash agreement (per ORT version)

### ORT 1.17.3  (AMD64 vs aarch64)

| Model | threads | AMD64 | aarch64 | verdict |
|---|---|---|---|---|
| clf_gb_clf | 1 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 2 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 4 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_rf_clf | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| reg_gradient_boosting | 1 | d1b666d94815 | d1b666d94815 | **MATCH** |
| reg_gradient_boosting | 2 | 9eed58f74e38 | 9eed58f74e38 | **MATCH** |
| reg_gradient_boosting | 4 | d4a62df077bc | d4a62df077bc | **MATCH** |
| reg_random_forest | 1 | 55ffe003aa11 | 55ffe003aa11 | **MATCH** |
| reg_random_forest | 2 | 3c79dd777f5c | 3c79dd777f5c | **MATCH** |
| reg_random_forest | 4 | b1d3632fe0f9 | b1d3632fe0f9 | **MATCH** |
| reg_svr | 1 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 2 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 4 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| scaler | 1 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 2 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 4 | 482d3916b942 | 482d3916b942 | **MATCH** |

### ORT 1.18.0  (AMD64 vs aarch64)

| Model | threads | AMD64 | aarch64 | verdict |
|---|---|---|---|---|
| clf_gb_clf | 1 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 2 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 4 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_rf_clf | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| reg_gradient_boosting | 1 | d1b666d94815 | d1b666d94815 | **MATCH** |
| reg_gradient_boosting | 2 | 9eed58f74e38 | 9eed58f74e38 | **MATCH** |
| reg_gradient_boosting | 4 | d4a62df077bc | d4a62df077bc | **MATCH** |
| reg_random_forest | 1 | 55ffe003aa11 | 55ffe003aa11 | **MATCH** |
| reg_random_forest | 2 | 3c79dd777f5c | 3c79dd777f5c | **MATCH** |
| reg_random_forest | 4 | b1d3632fe0f9 | b1d3632fe0f9 | **MATCH** |
| reg_svr | 1 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 2 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 4 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| scaler | 1 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 2 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 4 | 482d3916b942 | 482d3916b942 | **MATCH** |

### ORT 1.19.2  (AMD64 vs aarch64)

| Model | threads | AMD64 | aarch64 | verdict |
|---|---|---|---|---|
| clf_gb_clf | 1 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 2 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 4 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_rf_clf | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| reg_gradient_boosting | 1 | d1b666d94815 | d1b666d94815 | **MATCH** |
| reg_gradient_boosting | 2 | 9eed58f74e38 | 9eed58f74e38 | **MATCH** |
| reg_gradient_boosting | 4 | d4a62df077bc | d4a62df077bc | **MATCH** |
| reg_random_forest | 1 | 55ffe003aa11 | 55ffe003aa11 | **MATCH** |
| reg_random_forest | 2 | 3c79dd777f5c | 3c79dd777f5c | **MATCH** |
| reg_random_forest | 4 | b1d3632fe0f9 | b1d3632fe0f9 | **MATCH** |
| reg_svr | 1 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 2 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 4 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| scaler | 1 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 2 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 4 | 482d3916b942 | 482d3916b942 | **MATCH** |

### ORT 1.20.1  (AMD64 vs aarch64)

| Model | threads | AMD64 | aarch64 | verdict |
|---|---|---|---|---|
| clf_gb_clf | 1 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 2 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 4 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_rf_clf | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| reg_gradient_boosting | 1 | d1b666d94815 | d1b666d94815 | **MATCH** |
| reg_gradient_boosting | 2 | 9eed58f74e38 | 9eed58f74e38 | **MATCH** |
| reg_gradient_boosting | 4 | d4a62df077bc | d4a62df077bc | **MATCH** |
| reg_random_forest | 1 | 55ffe003aa11 | 55ffe003aa11 | **MATCH** |
| reg_random_forest | 2 | 3c79dd777f5c | 3c79dd777f5c | **MATCH** |
| reg_random_forest | 4 | b1d3632fe0f9 | b1d3632fe0f9 | **MATCH** |
| reg_svr | 1 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 2 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 4 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| scaler | 1 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 2 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 4 | 482d3916b942 | 482d3916b942 | **MATCH** |

### ORT 1.22.0  (AMD64 vs aarch64)

| Model | threads | AMD64 | aarch64 | verdict |
|---|---|---|---|---|
| clf_gb_clf | 1 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 2 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_gb_clf | 4 | d16546211cf9 | d16546211cf9 | **MATCH** |
| clf_rf_clf | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_rf_clf | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 1 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 2 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| clf_svc | 4 | ae1fd128caf8 | ae1fd128caf8 | **MATCH** |
| reg_gradient_boosting | 1 | d1b666d94815 | d1b666d94815 | **MATCH** |
| reg_gradient_boosting | 2 | 9eed58f74e38 | 9eed58f74e38 | **MATCH** |
| reg_gradient_boosting | 4 | d4a62df077bc | d4a62df077bc | **MATCH** |
| reg_random_forest | 1 | 55ffe003aa11 | 55ffe003aa11 | **MATCH** |
| reg_random_forest | 2 | 3c79dd777f5c | 3c79dd777f5c | **MATCH** |
| reg_random_forest | 4 | b1d3632fe0f9 | b1d3632fe0f9 | **MATCH** |
| reg_svr | 1 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 2 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| reg_svr | 4 | b7d5c5be81fa | d01eff08cd0b | **DIFFER** |
| scaler | 1 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 2 | 482d3916b942 | 482d3916b942 | **MATCH** |
| scaler | 4 | 482d3916b942 | 482d3916b942 | **MATCH** |

## Summary

- cells MATCH across arch: **90**
- cells DIFFER across arch: **15**

> A non-zero DIFFER count means ONNX inference is **not** bitwise-identical across CPU architectures for those cells; the determinism claim must be scoped per-architecture (the producing arch is anchored on-chain alongside ORT+threads).