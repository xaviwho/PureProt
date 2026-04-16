## Table V-F.1: Bitwise Reproducibility (40 runs, synthetic 200-sample input)

| Model                 |   Runs |   Unique output hashes | Bitwise identical   |   Inference latency (ms) |
|:----------------------|-------:|-----------------------:|:--------------------|-------------------------:|
| clf_gb_clf            |     40 |                      1 | YES                 |                      0.5 |
| clf_rf_clf            |     40 |                      1 | YES                 |                      1   |
| clf_svc               |     40 |                      1 | YES                 |                    742   |
| reg_gradient_boosting |     40 |                      1 | YES                 |                      0.3 |
| reg_random_forest     |     40 |                      1 | YES                 |                      1.4 |
| reg_svr               |     40 |                      1 | YES                 |                   1199   |
| scaler                |     40 |                      1 | YES                 |                      1.9 |

## Table V-F.2: sklearn ↔ ONNX Concordance

| Model                 | Concordance metric     | Within tolerance   |
|:----------------------|:-----------------------|:-------------------|
| clf_gb_clf            | 100.0% label agreement | YES                |
| clf_rf_clf            | 100.0% label agreement | YES                |
| clf_svc               | 100.0% label agreement | YES                |
| reg_gradient_boosting | max_diff = 8.12e-07    | YES                |
| reg_random_forest     | max_diff = 1.92e-06    | YES                |
| reg_svr               | max_diff = 2.12e-04    | NO                 |
| scaler                | max_diff = 1.53e-05    | YES                |

**Summary:** 7/7 models bitwise-deterministic; 6/7 models within concordance tolerance.

**SVR exception:** max_diff = 2.12×10⁻⁴ between sklearn (float64) and ONNX (float32) predictions, driven by RBF-kernel exp() precision loss in float32. This is reproducible and characterised rather than a determinism failure -- the ONNX model itself is bitwise identical across 40 runs.
