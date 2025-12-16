# PureProtX: Modular CLI Protocol for Blockchain-Audited Consensus AI and Docking-Based Virtual Screening

## Overview

PureProtX is a truly modular command-line interface (CLI) system that delivers on the promise of transparent, reproducible drug discovery with comprehensive blockchain auditing. Unlike traditional single-model approaches, PureProtX implements a **Consensus AI** system using ensemble methods (SVR + Random Forest + Gradient Boosting) for superior predictive accuracy.

This system is designed for researchers who demand rigorous reproducibility and regulatory compliance. Every aspect of the screening process - from AI model files to protein structures and parameters - is cryptographically hashed and recorded on the Purechain blockchain, creating an unassailable audit trail.

## Key Features

### **Modular Architecture**
- **AI Module**: Consensus AI with ensemble of SVR, Random Forest, and Gradient Boosting
- **Docking Module**: Advanced molecular docking with multiple engine support
- **Blockchain Module**: Comprehensive audit trail covering all components
- **Data Module**: Seamless ChEMBL integration and dataset management

### **Consensus AI System**
- **True Ensemble**: Three different ML algorithms working in consensus
- **Superior Accuracy**: Mathematically proven better performance than individual models
- **Individual Tracking**: Monitor performance of each model in the ensemble
- **Robust Predictions**: Reduced variance through ensemble averaging

### **Comprehensive Blockchain Auditing**
- **Model Hashing**: SHA-256 hashes of AI model files for reproducibility
- **Protein Hashing**: Cryptographic verification of protein structure files
- **Parameter Tracking**: Complete audit of all screening parameters
- **Zero Gas Fees**: Leverages Purechain network for cost-effective verification
- **Regulatory Compliance**: Immutable audit trail for regulatory submissions

### **Advanced Screening Capabilities**
- **Hybrid Screening**: AI + molecular docking with consensus scoring
- **Batch Processing**: High-throughput screening with comprehensive auditing
- **Real-time Verification**: Instant blockchain verification of results
- **Multi-target Support**: Pre-trained models for HIV-1 protease, BRAF, EGFR

## Project Structure

```
.
├── pureprot/                       # MODULAR CORE COMPONENTS
│   ├── __init__.py                 # Package initialization
│   ├── ai_model.py                 # Consensus AI Module (SVR+RF+GB)
│   ├── blockchain.py               # Comprehensive Blockchain Auditor
│   ├── docking.py                  # Advanced Docking Engine
│   └── data.py                     # Data Management Module
├── blockchain/                     # Legacy blockchain components
│   ├── purechain_connector.py      # Web3 blockchain connector
│   └── DrugScreeningVerifier.sol   # Smart contract for verification
├── modeling/                       # Legacy modeling components
│   ├── data_loader.py              # ChEMBL data fetching
│   ├── model_trainer.py            # Model training utilities
│   ├── molecular_modeling.py       # RDKit molecular modeling
│   └── advanced_docking_engine.py  # Multi-engine docking support
├── workflow/
│   └── verification_workflow.py    # Legacy verification workflow
├── docs/                           # Documentation
│   ├── technical_guide.md          # Technical implementation guide
│   └── Windows_Native_Docking_Guide.md # Windows docking setup
├── PureProt.py                     # Main CLI entry point
└── README.md                       # This file
```

### The "X" in PureProtX

The **"X"** represents the **eXtended** and **eXperimental** evolution from the original PureProt system:
- **eXtended**: True modular architecture with interchangeable components
- **eXperimental**: Cutting-edge Consensus AI and comprehensive blockchain auditing
- **eXcellence**: Superior performance through ensemble methods and rigorous verification

## Installation

1.  **Clone the repository**

2.  **Create a Python virtual environment**:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your blockchain wallet securely**:
    
    **Option A: Interactive Setup (Recommended)**
    ```bash
    python setup_env.py
    ```
    This will guide you through secure environment setup.
    
    **Option B: Manual Setup**
    Copy the example file and edit it:
    ```bash
    cp .env.example .env
    # Edit .env with your actual private key
    ```
    
    **Security Requirements:**
    - `.env` file is already in `.gitignore` - NEVER commit it
    - Your private key is used for blockchain transactions
    - Purechain network has zero gas fees
    - Keep your private key secure and never share it

## A Full Workflow Example

Here is how you can use PureProt to perform a complete, end-to-end virtual screening:

### Step 1: Fetch Data for a Target

First, download and prepare training data for a specific biological target from ChEMBL. We'll use BRAF (CHEMBL4822) as an example.

```bash
python PureProt.py fetch-data "CHEMBL4822" --output "braf_data.csv"
```

This command creates `braf_data.csv`, a file containing molecules and their known pIC50 values for the BRAF target.

### Step 2: Train a Custom AI Model

Next, train a new AI model on the data you just downloaded.

```bash
python PureProt.py train-model "braf_data.csv" --output "braf_model.joblib"
```

This creates `braf_model.joblib`, a trained model file ready for screening.

### Step 3: Screen a Molecule

Now, use your custom-trained model to screen a new molecule. The result will be automatically recorded on the blockchain.

```bash
python PureProt.py screen "MyBrafTest-01" --smiles "CNC(=O)c1cc(c(cn1)Oc1ccc(cc1)F)NC(=O)C(C)(C)C" --model "braf_model.joblib"
```

Take note of the `job_id` returned in the output.

### Step 4: Verify the Result

Finally, use the `job_id` to verify that the result stored locally matches the immutable record on the blockchain.

```bash
python PureProt.py verify "<your_job_id_from_step_3>"
```

A successful verification will return `"verified": true`.

### Step 5: View Job History

You can view a summary of all your past screening jobs at any time:

```bash
python PureProt.py history
```

## Molecular Docking Workflow

PureProtX provides comprehensive molecular docking capabilities with blockchain audit trails:

### Step 1: Prepare Protein Structure

Convert your protein PDB file to PDBQT format for docking:

```bash
python PureProt.py prep-protein "1hpv.pdb" --output "1hpv_prepared.pdbqt"
```

### Step 2: Find Binding Site

Automatically detect the binding site coordinates:

```bash
python PureProt.py find-binding-site "1hpv_prepared.pdbqt"
```

This will output the center coordinates and suggested box size for docking.

### Step 3: Single Molecule Docking

Dock a single molecule with comprehensive parameter tracking:

```bash
python PureProt.py dock "aspirin" \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --receptor "1hpv_prepared.pdbqt" \
  --center 10.0 15.0 20.0 \
  --size 20.0 20.0 20.0 \
  --exhaustiveness 8 \
  --output "aspirin_docking.csv"
```

### Step 4: Batch Docking

Dock multiple molecules from a CSV file:

```bash
python PureProt.py dock-batch "molecules.csv" \
  --receptor "1hpv_prepared.pdbqt" \
  --center 10.0 15.0 20.0 \
  --size 20.0 20.0 20.0 \
  --exhaustiveness 8 \
  --output "batch_docking_scores.csv" \
  --limit 100
```

**Input CSV format:**
```csv
molecule_id,smiles
aspirin,CC(=O)OC1=CC=CC=C1C(=O)O
ibuprofen,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
paracetamol,CC(=O)NC1=CC=C(C=C1)O
```

### Step 5: Hybrid AI+Docking Screening

Combine consensus AI predictions with molecular docking:

```bash
python PureProt.py hybrid-screen "molecules.csv" \
  --model "braf_consensus_model.joblib" \
  --protein "1hpv_prepared.pdbqt" \
  --center "10.0,15.0,20.0" \
  --size "20.0,20.0,20.0"
```

### Blockchain Audit Features

All docking operations include comprehensive blockchain auditing:

- **Receptor File Hash**: SHA-256 of the PDBQT structure
- **Parameter Tracking**: Center coordinates, box size, exhaustiveness
- **Result Verification**: Immutable docking scores and poses
- **Deterministic JSON**: All results appended to `pureprot_deterministic_results.json`
- **Master Hash**: Complete audit trail for reproducibility

## CLI Command Reference

### Core Commands
-   `info`: Displays project information and command usage.
-   `connect`: Tests the connection to the Purechain blockchain.
-   `fetch-data <target_id>`: Fetches and prepares data for a ChEMBL target.
-   `train-model <dataset_path>`: Trains a Consensus AI model (SVR+RF+GB ensemble).

### Screening Commands
-   `screen <molecule_id>`: Screens a single molecule with Consensus AI.
-   `batch <csv_path>`: Screens a batch of molecules with comprehensive audit.

### Docking Commands
-   `prep-protein <pdb_path>`: Prepares protein structure for docking (PDB → PDBQT).
-   `find-binding-site <protein_path>`: Auto-detects binding site coordinates.
-   `dock <molecule_id>`: Docks single molecule with blockchain audit.
-   `dock-batch <csv_path>`: Docks multiple molecules from CSV with audit.
-   `hybrid-screen <csv_path>`: Hybrid AI+docking screening with consensus scoring.

### Verification Commands
-   `verify <job_id>`: Verifies a screening result from the blockchain.
-   `history`: Shows the history of all screening jobs.

For more details on any command, run `python PureProt.py [command] --help`.
- Performance optimizations for batch screening

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

MIT License

## Acknowledgements

- This project was developed for APCC conference submission
- Thanks to Purechain for providing blockchain infrastructure
