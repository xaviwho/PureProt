import json, numpy as np, os
for f in sorted(os.listdir('docking_cache')):
    if not f.endswith('.json'):
        continue
    with open(f'docking_cache/{f}') as fh:
        d = json.load(fh)
    scores = np.array(d['scores'])
    nz = scores != 0.0
    valid = scores[nz]
    if len(valid) == 0:
        print(f'{f:40s}  NO VALID SCORES')
        continue
    outliers = int(np.sum(np.abs(valid) > 20))
    print(f'{f:40s}  valid={len(valid):5d}  mean={np.mean(valid):8.2f}  '
          f'min={np.min(valid):8.2f}  max={np.max(valid):10.2f}  |>20|: {outliers}')
