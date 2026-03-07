# PureProtX: Blockchain-Audited Consensus AI for Virtual Screening

A modular CLI system for reproducible, blockchain-audited drug virtual screening. PureProtX fuses a 7-model consensus AI ensemble with AutoDock Vina docking scores via a per-target alpha parameter, and commits every pipeline stage hash to PureChain for tamper-proof provenance.

## Quick Start (Reproduce Paper Results)

### Prerequisites

- **Python 3.12** (3.14 is not supported due to rdkit-pypi compatibility)
- **Conda** (recommended) or pip
- **AutoDock Vina** (included in `tools/` for Windows; Linux users install via Dockerfile)
- **Git**

### 1. Clone and set up environment

```bash
git clone https://github.com/xaviwho/PureProt.git
cd PureProt
```

**Option A: Conda (recommended)**

```bash
conda env create -f environment.yml
conda activate pureprotx
```

**Option B: pip**

```bash
python3.12 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Option C: Docker**

```bash
docker build -t pureprotx .
docker run -it pureprotx
```

### 2. Blockchain wallet setup

A pre-configured test wallet with a pre-funded PureChain account is included in the repository (`.env.example`) for immediate reproducibility. PureChain is a zero-gas-fee PoA network, so no cryptocurrency purchase is required.

**For immediate use (test wallet):**

```bash
cp .env.example .env
```

Then edit `.env` and set `TEST_PRIVATE_KEY` to a valid PureChain private key. The repository `.env.example` contains a placeholder; the actual test key used for the paper experiments is documented in the data availability statement.

**To generate your own wallet:**

```python
from web3 import Web3
acct = Web3().eth.account.create()
print(f"Address:     {acct.address}")
print(f"Private key: {acct.key.hex()}")
# Paste the private key (without 0x) into .env as TEST_PRIVATE_KEY
```

PureChain accounts are funded automatically on first use (zero-fee network). No faucet or token purchase is needed.

**Interactive setup (alternative):**

```bash
python setup_env.py
```

### 3. Reproduce the 10-target benchmark

```bash
python experiments/run_experiments.py
```

This runs all 7 experiments from the paper:

| Exp | Description | Output |
|-----|-------------|--------|
| 1 | Model training (4 reg + 3 clf) across 10 targets | R^2, RMSE, AUC-ROC per target |
| 2 | Enrichment metrics (EF@k%, BEDROC, Brier) | Enrichment table |
| 3 | Alpha optimisation + exhaustiveness sensitivity | Per-target alpha, Spearman rho |
| 4 | Determinism validation (9/9 components) | PASS/FAIL per component |
| 5 | Tamper detection and blockchain provenance | On-chain tx hashes |
| 6 | Scaffold diversity analysis | Novel fraction per target |
| 7 | DUD-E external benchmark (5 targets) | Transfer BEDROC |

Results are written to `experiments/paper_results/`:
- `revised_results.json` -- full structured results
- `revised_summary.md` -- human-readable summary
- `review_analyses.json` -- exhaustiveness sensitivity and chemical space overlap
- `figures/` -- all paper figures (PNG, 300 DPI)

### 4. Regenerate figures

```bash
python scripts/generate_figures.py
```

Produces single-panel, publication-quality figures in `experiments/paper_results/figures/`.

## 10-Target Benchmark Panel

8 protein families, 10 targets (including 3 HIV-1 antiviral targets covering distinct replication stages):

| ChEMBL ID | Target | Family | N compounds |
|-----------|--------|--------|-------------|
| CHEMBL243 | HIV-1 Protease | Viral protease | 3,444 |
| CHEMBL247 | HIV-1 Reverse Transcriptase | Viral polymerase | 10,308 |
| CHEMBL3471 | HIV-1 Integrase | Viral integrase | 7,879 |
| CHEMBL279 | VEGFR2 (KDR) | Kinase | 14,008 |
| CHEMBL2487 | Amyloid-beta A4 (APP) | Membrane protein | 999 |
| CHEMBL251 | Adenosine A2a receptor | GPCR | 2,126 |
| CHEMBL217 | Dopamine D2 receptor | GPCR | 1,570 |
| CHEMBL1862 | Estrogen Receptor alpha | Nuclear receptor | 5,156 |
| CHEMBL4005 | PPARgamma | Nuclear receptor | 9,723 |
| CHEMBL240 | hERG | Ion channel | 16,640 |

## Consensus AI Architecture

The ensemble comprises **4 regression models** and **3 classification models** (4+3 architecture):

- **Regression**: SVR, Random Forest, Gradient Boosting, MLP
- **Classification**: SVC, RF, GB
- **Features**: 2,048-bit Morgan fingerprints (radius=2) + 10 physicochemical descriptors = 2,058 dimensions
- **Consensus**: arithmetic mean of individual model predictions

Hyperparameters are tuned on the validation set (no test leakage). SVR/SVC uses subsampling to 3,000 compounds for datasets exceeding that size.

## Hybrid Scoring

```
f_hybrid(x) = alpha * f_AI(x) + (1 - alpha) * f_dock(x)
```

- `f_AI`: z-normalised consensus regression score
- `f_dock`: z-normalised AutoDock Vina binding affinity (exhaustiveness=4)
- `alpha`: optimised per-target on validation BEDROC; mean 0.895, LOTO 0.935

Docking scores are real Vina scores cached in `docking_cache/` (both e=1 and e=4 variants).

## Project Structure

```
PureProt/
├── pureprot/                          # Core library
│   ├── ai_model.py                    # ConsensusAIModel (4+3 ensemble, HP tuning)
│   ├── blockchain.py                  # BlockchainAuditor (hash, record, verify)
│   ├── docking.py                     # DockingEngine wrapper
│   ├── data.py                        # DataManager (ChEMBL fetch, splits)
│   ├── evaluation.py                  # Normalisation, BEDROC, EF, Brier, LOTO alpha
│   ├── ranking.py                     # Hybrid ranking and alpha grid search
│   ├── scaffold.py                    # Bemis-Murcko scaffold diversity analysis
│   └── targets.py                     # 10-target metadata and split configuration
├── blockchain/
│   ├── purechain_connector.py         # Web3 PureChain connector (zero-gas PoA)
│   ├── DrugScreeningVerifier.sol      # Solidity verification contract
│   └── deploy.py                      # Contract deployment script
├── experiments/
│   ├── run_experiments.py             # Main experiment runner (7 experiments)
│   └── paper_results/                 # Output directory
│       ├── revised_results.json       # Structured results
│       ├── review_analyses.json       # Exhaustiveness + chemical space overlap
│       ├── figures/                   # Publication figures (PNG + LaTeX tables)
│       └── models/                    # Trained model files (not in git, ~1.5 GB)
├── scripts/
│   ├── generate_figures.py            # Figure generation (single-panel, 300 DPI)
│   ├── run_vina_docking.py            # ChEMBL batch Vina docking (e=4)
│   ├── run_dude_docking.py            # DUD-E batch Vina docking (e=4)
│   └── analyze_review_issues.py       # DUD-E overlap + exhaustiveness analysis
├── docking_cache/                     # Vina score caches (not in git)
│   ├── CHEMBL243_vina_e4.json         # Per-target e=4 caches
│   └── ...
├── dude_data/                         # DUD-E actives/decoys (5 targets)
├── structures/                        # PDB/PDBQT protein structures
├── tools/                             # AutoDock Vina binaries (Windows)
├── docs/
│   └── methodology.md                 # Full methodology for all 7 experiments
├── PureProt.py                        # CLI entry point
├── Dockerfile                         # Reproducible container build
├── environment.yml                    # Conda environment specification
├── requirements.txt                   # pip requirements
├── setup_env.py                       # Interactive wallet setup
├── .env.example                       # Template for blockchain credentials
└── .env                               # Local credentials (gitignored)
```

## CLI Usage

### End-to-end screening workflow

```bash
# 1. Fetch ChEMBL bioactivity data for a target
python PureProt.py fetch-data "CHEMBL243" --output "hiv1pr_data.csv"

# 2. Train a consensus AI model
python PureProt.py train-model "hiv1pr_data.csv" --output "hiv1pr_model.joblib"

# 3. Screen a molecule (result is blockchain-audited)
python PureProt.py screen "test-mol-01" \
  --smiles "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(O)C(CC1=CC=CC=C1)NC(=O)C(CC(N)=O)NC(=O)C1=CC2=CC=CC=C2N1" \
  --model "hiv1pr_model.joblib"

# 4. Verify the result against the blockchain
python PureProt.py verify "<job_id>"

# 5. View screening history
python PureProt.py history
```

### Molecular docking

```bash
# Prepare protein
python PureProt.py prep-protein "structures/1hpv.pdb" --output "1hpv.pdbqt"

# Find binding site
python PureProt.py find-binding-site "1hpv.pdbqt"

# Dock a single molecule
python PureProt.py dock "aspirin" \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --receptor "1hpv.pdbqt" \
  --center 10.0 15.0 20.0 \
  --size 20.0 20.0 20.0 \
  --exhaustiveness 4

# Batch docking from CSV
python PureProt.py dock-batch "molecules.csv" \
  --receptor "1hpv.pdbqt" \
  --center 10.0 15.0 20.0 \
  --size 20.0 20.0 20.0

# Hybrid AI + docking screening
python PureProt.py hybrid-screen "molecules.csv" \
  --model "hiv1pr_model.joblib" \
  --protein "1hpv.pdbqt" \
  --center "10.0,15.0,20.0" \
  --size "20.0,20.0,20.0"
```

### Blockchain operations

```bash
# Test PureChain connection
python PureProt.py connect

# View system info
python PureProt.py info
```

## Blockchain Provenance

All screening results are hashed and committed to **PureChain** (Chain ID 900520900520), a zero-fee Proof-of-Authority network. The audit trail includes:

- **Merkle tree**: 4 pipeline stage hashes (fetch, train, dock, score) combined into a single root
- **Tamper detection**: any post-hoc modification produces a different hash, detectable on-chain
- **Selective verification**: verify any single stage without re-running the full pipeline

| Property | PureChain |
|----------|-----------|
| Consensus | Proof of Authority |
| Gas fees | Zero |
| Chain ID | 900520900520 |
| RPC endpoint | `https://purechainnode.com:8547` |
| Currency | PCC |

## Reproducibility

| Component | Version |
|-----------|---------|
| Python | 3.12.10 |
| scikit-learn | 1.8.0 |
| RDKit | 2025.09.4 |
| Web3.py | 7.14.1 |
| AutoDock Vina | 1.2.5 (Windows) / 1.2.7 (Docker) |

All random seeds fixed at `random_state=42`. Determinism validated: 9/9 components PASS, 100% hash reproducibility across 40 re-executions.

## License

MIT License

## Acknowledgements

- PureChain for providing zero-fee blockchain infrastructure
- ChEMBL for open bioactivity data
- DUD-E for external validation benchmarks
