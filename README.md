# PureProt: An AI-Blockchain Enabled Virtual Screening Tool for Drug Discovery

## Overview

PureProt is a command-line interface (CLI) tool designed to provide a transparent, reproducible, and user-friendly virtual screening workflow for drug discovery. It seamlessly integrates automated data fetching, AI model training, molecular screening, and blockchain-based verification to create a complete, end-to-end scientific pipeline.

This tool is built for researchers who need to perform virtual screenings and wish to maintain a verifiable, immutable record of their results. By leveraging the Purechain blockchain, every screening job can be recorded and later verified, ensuring the integrity and auditability of the scientific process.

## Features

- **End-to-End Workflow**: A complete pipeline from data acquisition to verifiable results.
- **Automated Data Fetching**: Download and prepare bioactivity data from the ChEMBL database with a single command.
- **Custom AI Model Training**: Train your own Support Vector Regression (SVR) models on custom datasets.
- **Dynamic Model Loading**: Screen molecules using either the default model or your own custom-trained models.
- **RDKit-Powered Screening**: Performs binding affinity prediction and calculates drug-like properties (Lipinski's Rule of Five) using RDKit.
- **Blockchain Verification**: Records a hash of each screening result on the Purechain blockchain, allowing anyone to verify the result's integrity.
- **Persistent Job History**: Automatically saves all screening results to `pureprot_results.json`, creating a stateful history of all your work.
- **User-Friendly CLI**: A simple and intuitive command-line interface makes the entire workflow accessible.

## Project Structure

```
.
├── blockchain/
│   ├── purechain_connector.py      # Handles all interaction with the Purechain blockchain.
│   └── DrugScreeningVerifier.sol   # The Solidity smart contract for on-chain verification.
├── modeling/
│   ├── data_loader.py              # Fetches and prepares data from ChEMBL.
│   ├── model_trainer.py            # Trains and saves the AI model.
│   └── molecular_modeling.py       # Core screening pipeline using RDKit and AI models.
├── workflow/
│   └── verification_workflow.py    # Integrates the AI screening and blockchain verification.
├── PureProt.py                     # The main CLI entry point.
├── pureprot_results.json           # Stores the history of all screening jobs (auto-generated).
├── requirements.txt                # Python dependencies.
└── README.md                       # This file.
```

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

4.  **Set up your blockchain wallet**:
    Create a `.env` file in the root directory and add your wallet's private key:
    ```
    TEST_PRIVATE_KEY="your_private_key_here"
    ```
    **Note**: This key is used to pay for gas fees when recording results on the blockchain. The Purechain testnet is configured for gas-free transactions.

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

## CLI Command Reference

-   `info`: Displays project information and command usage.
-   `connect`: Tests the connection to the Purechain blockchain.
-   `fetch-data <target_id>`: Fetches and prepares data for a ChEMBL target.
-   `train-model <dataset_path>`: Trains a new model on a dataset.
-   `screen <molecule_id>`: Screens a single molecule.
-   `batch <csv_path>`: Screens a batch of molecules from a CSV file.
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
