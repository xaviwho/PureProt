| Model | Within-config (40 runs) | Cross-thread {1,2,4} | Cross-ORT (1.17-1.22) | dtype residual vs sklearn |
|---|---|---|---|---|
| SVR (reg) | identical | stable | stable | f32 1.25e-04; f64 n/a^\dagger |
| RandomForest (reg) | identical | varies | stable | f64 n/a^\dagger |
| GradBoosting (reg) | identical | varies | stable | f64 n/a^\dagger |
| SVC (clf) | identical | stable | stable | -- |
| RandomForest (clf) | identical | stable | stable | -- |
| GradBoosting (clf) | identical | stable | stable | -- |
| StandardScaler | identical | stable | stable | f32 1.65e-05; f64 exact (0) |