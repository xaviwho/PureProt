# Technical Guide: Verifiable Drug Screening with Real Data

## Overview

This guide provides technical details on implementing and running the Verifiable Virtual Drug Screening System with real molecular data and blockchain verification.

## System Requirements

### Software Dependencies

- Python 3.8+
- RDKit (`conda install -c conda-forge rdkit` or `pip install rdkit`)
- Web3.py (`pip install web3`)
- NumPy (`pip install numpy`)
- Additional libraries: hashlib, json, time, typing, unittest, getpass

### Blockchain Requirements

- Access to Purechain network (RPC URL: `http://43.200.53.250:8548`)
- MetaMask wallet or private key
- PCC cryptocurrency for transaction fees

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PureProt.git
   cd PureProt
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install RDKit with Conda for better performance:
   ```bash
   conda install -c conda-forge rdkit
   ```

## System Architecture

### Directory Structure

```
/PureProt
├── blockchain/         # Blockchain interaction modules
├── modeling/          # Molecular modeling and prediction
├── workflow/          # Main workflow orchestration
├── tests/             # System tests
├── docs/              # Documentation
├── requirements.txt   # Python dependencies
└── README.md          # Project overview
```

### Key Components

1. **Blockchain Connector** (`blockchain/purechain_connector.py`)
   - Connects to Purechain blockchain
   - Handles transaction creation, signing, and verification
   - Manages wallet connections

2. **Molecular Modeling** (`modeling/molecular_modeling.py`)
   - Creates molecular representations from SMILES strings
   - Extracts features using RDKit
   - Calculates molecular fingerprints and hashes

3. **Prediction Models** (`modeling/binding_model.py`, `modeling/toxicity_model.py`)
   - Predicts binding affinities and toxicity scores
   - Uses real data for known compounds
   - Uses ML models trained on real data for novel compounds

4. **Verification Workflow** (`workflow/verification_workflow.py`)
   - Orchestrates the screening process
   - Records results on blockchain
   - Verifies results against blockchain records

## Usage Examples

### Basic Screening

```python
from workflow.verification_workflow import VerifiableDrugScreening

# Initialize workflow with Purechain RPC URL
workflow = VerifiableDrugScreening("http://43.200.53.250:8548")

# Connect to wallet
workflow.connect_wallet()

# Run screening for a molecule using its SMILES string
aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
result = workflow.run_screening("aspirin", aspirin_smiles)

# Record result on blockchain (returns transaction hash)
tx_hash = workflow.record_result(result)

# Later, verify the result
verified = workflow.verify_screening("aspirin", tx_hash)
```

### Batch Screening

```python
from workflow.verification_workflow import VerifiableDrugScreening

# Initialize workflow
workflow = VerifiableDrugScreening("http://43.200.53.250:8548")
workflow.connect_wallet()

# Define molecules with their SMILES
molecules = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "paracetamol": "CC(=O)NC1=CC=C(C=C1)O"
}

# Run batch screening
results = workflow.batch_screen_molecules(molecules)

# Record all results
tx_hashes = {}
for mol_id, result in results.items():
    tx_hashes[mol_id] = workflow.record_result(result)

# Save results and transaction hashes
workflow.save_results("screening_results.json")
```

## Integrating RDKit

The system uses RDKit to calculate real molecular properties:

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# Create molecule from SMILES
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
mol = Chem.MolFromSmiles(smiles)

# Calculate properties
mol_weight = Descriptors.MolWt(mol)                  # 180.16 g/mol
logp = Descriptors.MolLogP(mol)                      # 1.19
h_donors = Lipinski.NumHDonors(mol)                  # 1
h_acceptors = Lipinski.NumHAcceptors(mol)            # 4
tpsa = Descriptors.TPSA(mol)                         # 63.6 Å²
rot_bonds = Descriptors.NumRotatableBonds(mol)       # 3

# Generate fingerprint
fingerprint = GetMorganFingerprintAsBitVect(mol, 2, nBits=16)
```

## Blockchain Verification Details

The system uses these steps for blockchain verification:

1. **Hash Calculation**:
   - Molecular features and screening results are serialized to JSON
   - SHA-256 hash is calculated on the JSON string

2. **Transaction Creation**:
   - Hash is included in transaction data
   - Transaction is signed with wallet's private key
   - Transaction is sent to Purechain network

3. **Verification**:
   - Transaction is retrieved using transaction hash
   - Stored hash is extracted from transaction data
   - Current result is hashed again for comparison
   - Verification succeeds if hashes match

## Troubleshooting

### Common Issues

1. **RDKit Import Error**:
   - Issue: `ImportError: No module named 'rdkit'`
   - Solution: Install RDKit with Conda: `conda install -c conda-forge rdkit`

2. **Web3 Connection Error**:
   - Issue: `ConnectionError: Could not connect to Purechain RPC`
   - Solution: Check network connection and RPC URL

3. **Transaction Failures**:
   - Issue: Transactions fail with "Insufficient funds"
   - Solution: Ensure wallet has enough PCC for gas fees

## Performance Optimization

- Use batch processing for multiple molecules
- Cache molecular features for commonly used molecules
- Optimize RDKit calls for performance-critical sections
- Consider using multithreading for parallel processing

## References

1. RDKit Documentation: https://www.rdkit.org/docs/
2. Web3.py Documentation: https://web3py.readthedocs.io/
3. Purechain Documentation: [Purechain website]
