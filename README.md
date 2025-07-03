# Verifiable Virtual Drug Screening with Blockchain & AI

## Overview

This project implements a verifiable drug screening system that integrates AI-enhanced molecular modeling with blockchain verification. The system allows researchers to perform virtual drug screening with transparent, reproducible, and immutable results secured by the Purechain blockchain.

## Features

- **Blockchain-Verified Results**: All screening results are hashed and stored on the Purechain blockchain for verification and auditability
- **AI-Enhanced Molecular Modeling**: Simulated molecular feature extraction, binding affinity prediction, and toxicity screening
- **Command-Line Interface**: Easy-to-use CLI for running screening jobs and verifying results
- **Verifiable Workflow**: Complete integration between off-chain AI computations and on-chain verification
- **Comprehensive Testing**: Unit and integration tests for all components

## Project Structure

```
.
├── blockchain/              # Blockchain interaction components
│   ├── purechain_connector.py            # Purechain blockchain connector
│   ├── deploy.py                         # Smart contract deployment script
│   └── DrugScreeningVerifier.sol         # Solidity smart contract
├── docs/                   # Project documentation
│   └── ai_modeling_pipeline.md           # Explanation of the AI pipeline
├── modeling/               # AI molecular modeling components 
│   └── molecular_modeling.py             # Molecular representation and AI models
├── workflow/               # Core workflow components
│   └── verification_workflow.py          # Integration of blockchain and modeling
├── tests/                  # Test suite
│   └── test_system.py                   # Unit and integration tests
├── main.py                 # CLI interface
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Set up a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The system provides a CLI interface with the following commands:

- **Connect to blockchain**:
  ```bash
  python main.py connect
  ```

- **Screen a single molecule**:
  ```bash
  python main.py screen aspirin "CC(=O)OC1=CC=CC=C1C(=O)O" --target "protein1"
  ```

- **Batch screen molecules from a file**:
  ```bash
  python main.py batch molecules.json
  ```

- **Verify a screening result**:
  ```bash
  python main.py verify <job_id> <tx_hash>
  ```

- **Show job history**:
  ```bash
  python main.py history
  ```

## Configuration

Purechain blockchain configuration is set in `main.py`:

- RPC URL: `http://43.200.53.250:8548`
- Chain ID: `900520900520`
- Currency: `PCC`

## Testing

Run the test suite:

```bash
python -m unittest tests/test_system.py
```

## Future Enhancements

- Integration with real molecular modeling libraries (RDKit, PyTorch)
- Web interface with MetaMask wallet support
- Smart contract deployment on Purechain mainnet
- Performance optimizations for batch screening

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

MIT License

## Acknowledgements

- This project was developed for APCC conference submission
- Thanks to Purechain for providing blockchain infrastructure
