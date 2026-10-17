"""Generate the revision's latency figures from archived benchmark CSVs.

Run: python scripts/generate_revision_latency_figures.py
This regenerates figures; it does NOT rerun benchmarks or contact PureChain.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/images'
AUDIT = ROOT / 'paper/figure_data/latency_plot_audit.json'


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return [{key: float(value) for key, value in row.items()
                 if key != 'crossover'} for row in csv.DictReader(stream)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    scalability_path = ROOT / 'results/scalability_benchmark.csv'
    local_path = ROOT / 'results/provenance_comparison.csv'
    provenance_path = ROOT / 'results/provenance_manifest.json'
    scaling = sorted(read_rows(scalability_path), key=lambda row: row['N'])
    local = read_rows(local_path)
    for row in scaling:
        n = row['N']
        require(n > 0 and n.is_integer(), 'Invalid batch size')
        for strategy in ('a', 'b'):
            total = row[f'strategy_{strategy}_total_s']
            per_record = row[f'strategy_{strategy}_per_record_ms']
            require(math.isfinite(total) and total > 0, 'Invalid elapsed time')
            require(math.isclose(total * 1000 / n, per_record, abs_tol=.01),
                    f'N={n}: total/per-record inconsistency')
        ratio = row['strategy_a_total_s'] / row['strategy_b_total_s']
        require(math.isclose(ratio, row['speedup_factor'], abs_tol=.01),
                f'N={n}: speedup inconsistency')
    require(len({row['N'] for row in scaling}) == len(scaling), 'Duplicate N')
    anchor = next(row for row in scaling if row['N'] == 100)
    baseline = next(row for row in local if row['N'] == 100)
    values = [baseline['merkle_root_ms_per_rec'], baseline['cid_write_ms_per_rec'],
              baseline['ed25519_write_ms_per_rec'],
              anchor['strategy_b_per_record_ms'], anchor['strategy_a_per_record_ms']]
    require(all(math.isfinite(value) and value > 0 for value in values),
            'Logarithmic plot requires positive finite timings')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300})
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), layout='constrained')
    n_values = [row['N'] for row in scaling]
    for strategy, label, color, marker in [
        ('a', 'Individual commits', '#9b2929', 'o'),
        ('b', 'Merkle batch', '#176b45', 's')]:
        ax.plot(n_values, [row[f'strategy_{strategy}_per_record_ms'] for row in scaling],
                label=label, color=color, marker=marker, linewidth=2, markersize=6)
    ax.set(xscale='log', yscale='log', xlabel='Batch size N (records)',
           ylabel='Anchoring latency (ms/record)', xlim=(8, 125), ylim=(10, 4000))
    ax.set_xticks(n_values)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([10, 100, 1000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which='major', alpha=.22)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, fontsize=10)
    fig.savefig(OUT / 'fig_ve_scalability.png')
    plt.close(fig)

    labels = ['Merkle root build', 'Local content address', 'Ed25519 signed log',
              'Clique batch (N=100)', 'Clique individual']
    fig, ax = plt.subplots(figsize=(7, 4.3), layout='constrained')
    positions = list(range(len(values)))
    # A logarithmic point plot avoids implying a zero baseline on a log axis.
    ax.scatter(values[:3], positions[:3], color='#2367a0', s=65,
               label='Local operations (best of 5)')
    ax.scatter(values[3:], positions[3:], color='#9b2929', marker='s', s=65,
               label='Archived chain measurements (one run)')
    for position, value in zip(positions, values):
        label = f'{value:,.2f}' if value >= 1 else f'{value:.4g}'
        ax.annotate(label, (value, position), xytext=(9, 0),
                    textcoords='offset points', va='center', fontsize=10)
    ax.set(xscale='log', xlabel='Write latency (ms/record)', xlim=(.0005, 18000),
           ylim=(4.7, -.6))
    ax.set_yticks(positions, labels)
    ax.grid(True, axis='x', which='major', alpha=.22)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9)
    fig.savefig(OUT / 'fig3_provenance_cost.png')
    plt.close(fig)

    metadata = json.loads(provenance_path.read_text(encoding='utf-8'))
    audit = {
        'generator': 'scripts/generate_revision_latency_figures.py',
        'scope': 'Regenerated from archived runs; no new benchmark or live-chain verification.',
        'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (scalability_path, local_path, provenance_path)},
        'checks': ['Positive finite times', 'Unique scalability batch sizes',
                   'Totals agree with per-record times within CSV rounding',
                   'Speedups agree with ratios within CSV rounding'],
        'scalability_plot_rows': scaling,
        'provenance_plot_N': 100,
        'provenance_plot_values_ms_per_record': dict(zip(labels, values)),
        'local_measurement_protocol': metadata,
        'limitations': ['Scalability has one archived run per batch size; no uncertainty bars.',
                        'Local timings are best-of-five from a separate measurement protocol.',
                        'Content addressing measures local CID calculation, not IPFS network I/O.',
                        'These checks validate plotted data, not original live transaction execution.'],
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print('Verified archived CSV arithmetic; generated both figures and', AUDIT.relative_to(ROOT))


if __name__ == '__main__':
    main()
