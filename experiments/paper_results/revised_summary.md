# PureProtX Revised Results -- Summary
**Run date:** 2026-02-25T12:04:12.497871
**rdkit available:** True

## Experiment 1: Model Training (60/20/20 Splits)
| Target | N | Train | Val | Test | Consensus R2 | Consensus RMSE | AUC-ROC | AUC-PR |
|--------|---|-------|-----|------|-------------|----------------|---------|--------|
| CHEMBL243 | 3444 | 2066 | 689 | 689 | 0.7256 | 0.8742 | 0.9426 | 0.9806 |
| CHEMBL247 | 10308 | 6184 | 2062 | 2062 | 0.5635 | 0.9245 | 0.8952 | 0.9064 |
| CHEMBL279 | 14008 | 8404 | 2802 | 2802 | 0.6368 | 0.7575 | 0.9024 | 0.9484 |
| CHEMBL3471 | 7879 | 4727 | 1576 | 1576 | 0.7037 | 0.8249 | 0.9533 | 0.9065 |
| CHEMBL2487 | 999 | 599 | 200 | 200 | 0.7672 | 0.6427 | 0.9465 | 0.9703 |
| CHEMBL251 | 2126 | 1275 | 425 | 426 | 0.6903 | 0.7453 | 0.9757 | 0.9963 |
| CHEMBL217 | 1570 | 942 | 314 | 314 | 0.6390 | 0.9626 | 0.9457 | 0.9469 |
| CHEMBL1862 | 5156 | 3093 | 1031 | 1032 | 0.8081 | 0.6960 | 0.9610 | 0.9864 |
| CHEMBL4005 | 9723 | 5833 | 1945 | 1945 | 0.7436 | 0.6316 | 0.9376 | 0.9641 |
| CHEMBL240 | 16640 | 9984 | 3328 | 3328 | 0.6409 | 0.5902 | 0.9040 | 0.7610 |

## Experiment 2: Enrichment Metrics (Test Set)
| Target | Method | EF@1% | EF@5% | EF@10% | BEDROC | AUC-ROC |
|--------|--------|-------|-------|--------|--------|---------|
| CHEMBL243 | Regression | 1.29 | 1.29 | 1.29 | 0.882 | 0.940 |
| CHEMBL243 | Classification | 1.29 | 1.29 | 1.29 | 0.877 | 0.943 |
| CHEMBL243 | Hybrid(a=0.75) | 1.29 | 1.29 | 1.29 | 0.882 | 0.943 |
| CHEMBL247 | Regression | 1.83 | 1.76 | 1.74 | 0.887 | 0.884 |
| CHEMBL247 | Classification | 1.83 | 1.79 | 1.77 | 0.916 | 0.895 |
| CHEMBL247 | Hybrid(a=1.00) | 1.83 | 1.76 | 1.74 | 0.887 | 0.884 |
| CHEMBL279 | Regression | 1.52 | 1.52 | 1.51 | 0.966 | 0.890 |
| CHEMBL279 | Classification | 1.47 | 1.51 | 1.51 | 0.952 | 0.902 |
| CHEMBL279 | Hybrid(a=0.85) | 1.52 | 1.51 | 1.50 | 0.958 | 0.883 |
| CHEMBL3471 | Regression | 3.43 | 3.34 | 3.27 | 0.922 | 0.935 |
| CHEMBL3471 | Classification | 3.43 | 3.38 | 3.36 | 0.948 | 0.953 |
| CHEMBL3471 | Hybrid(a=1.00) | 3.43 | 3.34 | 3.27 | 0.922 | 0.935 |
| CHEMBL2487 | Regression | 1.44 | 1.44 | 1.44 | 0.742 | 0.944 |
| CHEMBL2487 | Classification | 1.44 | 1.44 | 1.44 | 0.720 | 0.946 |
| CHEMBL2487 | Hybrid(a=1.00) | 1.44 | 1.44 | 1.44 | 0.742 | 0.944 |
| CHEMBL251 | Regression | 1.15 | 1.15 | 1.15 | 0.735 | 0.924 |
| CHEMBL251 | Classification | 1.15 | 1.15 | 1.15 | 0.736 | 0.976 |
| CHEMBL251 | Hybrid(a=1.00) | 1.15 | 1.15 | 1.15 | 0.735 | 0.924 |
| CHEMBL217 | Regression | 1.95 | 1.95 | 1.89 | 0.848 | 0.943 |
| CHEMBL217 | Classification | 1.95 | 1.95 | 1.89 | 0.851 | 0.946 |
| CHEMBL217 | Hybrid(a=0.70) | 1.95 | 1.83 | 1.89 | 0.792 | 0.904 |
| CHEMBL1862 | Regression | 1.30 | 1.30 | 1.29 | 0.909 | 0.957 |
| CHEMBL1862 | Classification | 1.30 | 1.27 | 1.29 | 0.863 | 0.961 |
| CHEMBL1862 | Hybrid(a=0.70) | 1.30 | 1.30 | 1.30 | 0.917 | 0.929 |
| CHEMBL4005 | Regression | 1.50 | 1.50 | 1.50 | 0.959 | 0.922 |
| CHEMBL4005 | Classification | 1.43 | 1.49 | 1.50 | 0.936 | 0.938 |
| CHEMBL4005 | Hybrid(a=1.00) | 1.50 | 1.50 | 1.50 | 0.959 | 0.922 |
| CHEMBL240 | Regression | 4.75 | 4.84 | 4.79 | 0.804 | 0.912 |
| CHEMBL240 | Classification | 5.57 | 5.11 | 4.77 | 0.833 | 0.904 |
| CHEMBL240 | Hybrid(a=0.95) | 4.75 | 4.91 | 4.79 | 0.805 | 0.913 |

## Experiment 3: Alpha Optimization
| Target | Optimal alpha | Val BEDROC | Docking |
|--------|-----------|-----------|---------|
| CHEMBL243 | 0.75 | 0.89 | vina |
| CHEMBL247 | 1.00 | 0.92 | vina |
| CHEMBL279 | 0.85 | 0.96 | vina |
| CHEMBL3471 | 1.00 | 0.92 | vina |
| CHEMBL2487 | 1.00 | 0.78 | vina |
| CHEMBL251 | 1.00 | 0.74 | vina |
| CHEMBL217 | 0.70 | 0.83 | vina |
| CHEMBL1862 | 0.70 | 0.91 | vina |
| CHEMBL4005 | 1.00 | 0.96 | vina |
| CHEMBL240 | 0.95 | 0.82 | vina |

**Per-target alpha mean:** 0.895 +/- 0.125
**LOTO cross-validated alpha:** 0.935 +/- 0.039

## Experiment 4: Determinism Validation
**All tests passed:** True
- regression_models/SVR: **PASS**
- regression_models/RandomForest: **PASS**
- regression_models/GradientBoosting: **PASS**
- regression_models/MLP: **PASS**
- classification_models/SVC: **PASS**
- classification_models/RF_clf: **PASS**
- classification_models/GB_clf: **PASS**
- hash_pipeline: **PASS**
- consensus_pipeline: **PASS**

## Experiment 5: Tamper Detection + Blockchain Provenance
- Tamper detected: **True**
- Original hash: `cf4cd0a825bc9cb24d14ea5fbd73f9b7...`
- Tampered hash: `f11f2952585b1c7db035308a6f433789...`
- Merkle root: `4dda11fa096df49506dddfd347c9438f...`
- **PureChain tamper tx:** `096cef7fe83423c74418cc1eab934d7f26746b815d1b30f233ff7e40549eebf9`
- **PureChain tamper block:** 897718
- **PureChain Merkle tx:** `2a028e2be3c6e825098553f5c1261a10d439a32ac1bd6fdb52f769c9d0956fbd`
- **PureChain Merkle block:** 897720
- Offline mode: False

## Experiment 6: Scaffold Diversity (Top 10%)
| Target | Method | Unique Scaffolds | Novel Fraction | Tanimoto Mean |
|--------|--------|-----------------|----------------|---------------|
| CHEMBL243 | Regression | 44 | 29.55% | 0.946 |
| CHEMBL243 | Classification | 37 | 24.32% | 0.958 |
| CHEMBL243 | Hybrid | 44 | 31.82% | 0.940 |
| CHEMBL247 | Regression | 66 | 7.58% | 0.942 |
| CHEMBL247 | Classification | 76 | 11.84% | 0.944 |
| CHEMBL247 | Hybrid | 66 | 7.58% | 0.942 |
| CHEMBL279 | Regression | 127 | 20.47% | 0.914 |
| CHEMBL279 | Classification | 133 | 25.56% | 0.904 |
| CHEMBL279 | Hybrid | 127 | 22.05% | 0.908 |
| CHEMBL3471 | Regression | 85 | 47.06% | 0.882 |
| CHEMBL3471 | Classification | 84 | 44.05% | 0.891 |
| CHEMBL3471 | Hybrid | 85 | 47.06% | 0.882 |
| CHEMBL2487 | Regression | 9 | 11.11% | 0.913 |
| CHEMBL2487 | Classification | 16 | 0.00% | 0.829 |
| CHEMBL2487 | Hybrid | 9 | 11.11% | 0.913 |
| CHEMBL251 | Regression | 22 | 27.27% | 0.889 |
| CHEMBL251 | Classification | 19 | 5.26% | 0.898 |
| CHEMBL251 | Hybrid | 22 | 27.27% | 0.889 |
| CHEMBL217 | Regression | 18 | 5.56% | 0.950 |
| CHEMBL217 | Classification | 23 | 8.70% | 0.882 |
| CHEMBL217 | Hybrid | 21 | 19.05% | 0.900 |
| CHEMBL1862 | Regression | 40 | 5.00% | 0.956 |
| CHEMBL1862 | Classification | 27 | 0.00% | 0.987 |
| CHEMBL1862 | Hybrid | 38 | 10.53% | 0.954 |
| CHEMBL4005 | Regression | 130 | 36.15% | 0.889 |
| CHEMBL4005 | Classification | 113 | 25.66% | 0.889 |
| CHEMBL4005 | Hybrid | 130 | 36.15% | 0.889 |
| CHEMBL240 | Regression | 157 | 28.03% | 0.902 |
| CHEMBL240 | Classification | 166 | 18.67% | 0.930 |
| CHEMBL240 | Hybrid | 156 | 27.56% | 0.904 |

## Experiment 7: DUD-E Secondary Benchmark
| DUD-E Target | ChEMBL | Actives | Decoys | Reg BEDROC | Clf BEDROC | Hyb BEDROC | alpha |
|-------------|--------|---------|--------|-----------|-----------|-----------|-------|
| aa2ar | CHEMBL251 | 482 | 2000 | 0.907 | 0.939 | 0.911 | 0.85 |
| esr1 | CHEMBL1862 | 383 | 2000 | 0.000 | 0.000 | 0.383 | 0.15 |
| hivpr | CHEMBL243 | 536 | 2000 | 0.906 | 0.927 | 0.908 | 0.95 |
| pparg | CHEMBL4005 | 484 | 2000 | 0.000 | 0.000 | N/A | 0.00 |
| vgfr2 | CHEMBL279 | 409 | 2000 | 0.983 | 0.982 | 0.983 | 0.95 |
