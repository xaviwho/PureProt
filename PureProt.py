"""Main CLI Interface for the PureProt System

This module provides a command-line interface for transparent and reproducible
drug discovery powered by AI and blockchain technology.
"""

import argparse
import csv
import json
import os
import sys
import time
import statistics
from typing import Dict, Any, List, Optional

from workflow.verification_workflow import VerifiableDrugScreening
from modeling.data_loader import fetch_and_prepare_data
from modeling.model_trainer import train_and_save_model

# Purechain configuration
PURECHAIN_RPC_URL = "https://purechainnode.com:8547"
PURECHAIN_CHAIN_ID = 900520900520
PURECHAIN_CURRENCY = "PCC"

class PureProtCLI:
    """Main class for the PureProt command-line interface."""

    def __init__(self, results_file="pureprot_results.json"):
        """Initialize the CLI, setting up argument parsers for all commands."""
        self.results_file = results_file
        self.parser = argparse.ArgumentParser(
            description="PureProt: AI-Blockchain Enabled Virtual Screening for Drug Discovery.",
            formatter_class=argparse.RawTextHelpFormatter
        )
        self.parser.add_argument("-v", "--version", action="version", version="PureProt 1.0")
        self.subparsers = self.parser.add_subparsers(dest="command", help="Available commands")

        # --- Info Command ---
        self.subparsers.add_parser("info", help="Display project information and command usage.")

        # --- Connect Command ---
        self.subparsers.add_parser("connect", help="Test the connection to the Purechain blockchain.")

        # --- Screen Command ---
        screen_parser = self.subparsers.add_parser("screen", help="Screen a single molecule.")
        screen_parser.add_argument("molecule_id", type=str, help="Identifier for the molecule (e.g., a ChEMBL ID).")
        screen_parser.add_argument("--smiles", type=str, help="SMILES string of the molecule.")
        screen_parser.add_argument("--model", type=str, help="Path to a custom-trained .joblib model file.")
        screen_parser.add_argument("--target_id", type=str, default="default", help="Target protein or assay ID.")

        # --- Batch Command ---
        batch_parser = self.subparsers.add_parser("batch", help="Screen a batch of molecules from a CSV file.")
        batch_parser.add_argument("csv_path", type=str, help="Path to the CSV file containing molecules.")
        batch_parser.add_argument("--model", type=str, help="Path to a custom-trained .joblib model file.")

        # --- Verify Command ---
        verify_parser = self.subparsers.add_parser("verify", help="Verify a screening result from the blockchain.")
        verify_parser.add_argument("job_id", type=str, help="The job ID of the screening to verify.")

        # --- History Command ---
        self.subparsers.add_parser("history", help="Show the history of screening jobs.")

        # --- Benchmark Command ---
        benchmark_parser = self.subparsers.add_parser("benchmark", help="Run a benchmark on a dataset.")
        benchmark_parser.add_argument("dataset_path", type=str, help="Path to the dataset CSV file.")
        benchmark_parser.add_argument("--limit", type=int, default=None, help="Limit the number of molecules to process.")

        # --- Fetch Data Command ---
        fetch_parser = self.subparsers.add_parser("fetch-data", help="Fetch and prepare bioactivity data from ChEMBL.")
        fetch_parser.add_argument("target_id", type=str, help="ChEMBL ID of the target protein (e.g., CHEMBL240).")
        fetch_parser.add_argument("--output", type=str, help="Path to save the prepared data CSV file.")

        # --- Train Model Command ---
        train_parser = self.subparsers.add_parser("train-model", help="Train a new model on a dataset.")
        train_parser.add_argument("dataset_path", type=str, help="Path to the prepared data CSV file.")
        train_parser.add_argument("--output", type=str, help="Path to save the trained model file (e.g., model.joblib).")

    def run(self):
        """Parse arguments and execute the corresponding command."""
        args = self.parser.parse_args()

        if args.command == "info":
            self.show_info()
        elif args.command == "connect":
            workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID)
            workflow.test_connection()
        elif args.command == "screen":
            self.run_screen(args.molecule_id, args.smiles, args.model)
        elif args.command == "batch":
            self.run_batch(args.csv_path, args.model)
        elif args.command == "verify":
            self.run_verify(args.job_id)
        elif args.command == "history":
            self.run_history()
        elif args.command == "benchmark":
            self.run_benchmark(args.dataset_path, args.limit)
        elif args.command == "fetch-data":
            self.run_fetch_data(args.target_id, args.output)
        elif args.command == "train-model":
            self.run_train_model(args.dataset_path, args.output)
        else:
            self.parser.print_help()

    def show_info(self):
        """Display the welcome message and guide for the user."""
        info_text = """
        ========================================================
        PureProt: AI-Blockchain Enabled Virtual Screening
        ========================================================

        Welcome to PureProt, a scientific tool for transparent and reproducible
        drug discovery powered by AI and blockchain technology.

        This tool allows you to perform a full virtual screening workflow:
        1. Fetch Data: Download and prepare bioactivity data for a specific target.
        2. Train Model: Train a predictive AI model on the prepared data.
        3. Screen Molecules: Use the trained model to perform virtual screening.
        4. Verify Results: Record and verify screening results on the Purechain blockchain.

        Available Commands:
        -------------------
        info        : Show this information message.
        connect     : Test the connection to the Purechain blockchain.
        fetch-data  : Fetch and prepare bioactivity data from ChEMBL for a given target ID.
        train-model : Train a new AI model on a prepared dataset.
        screen      : Screen a single molecule by its ID or SMILES string.
        batch       : Screen a batch of molecules from a CSV file.
        verify      : Verify a past screening job using its job_id.
        history     : Display the history of all screening jobs.
        benchmark   : Run a performance and reliability benchmark on a dataset.

        Example Usage:
        --------------
        # Get information about the tool
        python PureProt.py info

        # Fetch data for a specific target
        python PureProt.py fetch-data "CHEMBL240" --output "braf_data.csv"

        # Train a model on the new data
        python PureProt.py train-model "braf_data.csv" --output "braf_model.joblib"

        # Screen a single molecule using the default model
        python PureProt.py screen "CHEMBL12345" --smiles "C1=CC=CC=C1"

        # Screen a single molecule using a custom-trained model
        python PureProt.py screen "CHEMBL12345" --smiles "C1=CC=CC=C1" --model "braf_model.joblib"

        # Run a benchmark on the first 10 molecules of a dataset
        python PureProt.py benchmark "path/to/your_data.csv" --limit 10

        For more details on a specific command, run:
        python PureProt.py [command] --help
        """
        print(info_text)

    def run_batch(self, csv_path: str, model_path: Optional[str] = None):
        """Run a batch screening job from a CSV file."""
        print(f"--- Batch Screening from: {csv_path} ---")
        workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID, model_path=model_path)
        workflow.load_results(self.results_file)  # Load existing history
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                molecules = list(reader)
        except Exception as e:
            print(f"Error: Could not read dataset file: {e}")
            return

        for i, row in enumerate(molecules):
            molecule_id = row.get('molecule_id') or row.get('canonical_smiles')
            smiles = row.get('smiles') or row.get('canonical_smiles')
            if not molecule_id or not smiles:
                print(f"Skipping row {i+1}: missing molecule identifier or SMILES string.")
                continue
            
            print(f"\n--- Processing Molecule {i+1}/{len(molecules)}: {molecule_id} ---")
            workflow.run_screening_job(molecule_id, smiles)
        
        workflow.save_results(self.results_file) # Save updated history
        print("\n--- Batch Screening Complete ---")

    def run_screen(self, molecule_id: str, smiles: Optional[str] = None, model_path: Optional[str] = None):
        """Run a single molecule screening job."""
        print(f"--- Screening Molecule: {molecule_id} ---")
        workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID, model_path=model_path)
        workflow.load_results(self.results_file)  # Load existing history
        result = workflow.run_screening_job(molecule_id=molecule_id, smiles=smiles)
        workflow.save_results(self.results_file) # Save updated history
        print("\n--- Screening Result ---")
        print(json.dumps(result, indent=4))

    def run_benchmark(self, dataset_path: str, limit: Optional[int] = None):
        """Run a full benchmark on a dataset, measuring performance and reliability."""
        print(f"--- Starting Benchmark ---")
        print(f"Dataset: {dataset_path}")
        workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID)
        workflow.run_benchmark(dataset_path, limit)

    def run_fetch_data(self, target_id: str, output_path: Optional[str] = None):
        """Fetch and prepare data for a given ChEMBL target ID."""
        print(f"--- Fetching Data for Target: {target_id} ---")
        if not output_path:
            output_path = f"{target_id.lower()}_prepared_data.csv"
            print(f"No output path specified. Saving to: {output_path}")
        
        fetch_and_prepare_data(target_id, output_path)

    def run_train_model(self, dataset_path: str, model_output_path: Optional[str] = None):
        """Train a new model from a dataset and save it."""
        print(f"--- Training Model from Dataset: {dataset_path} ---")
        if not model_output_path:
            model_output_path = "trained_model.joblib"
            print(f"No output path specified. Saving model to: {model_output_path}")
        
        train_and_save_model(dataset_path, model_output_path)


    def run_verify(self, job_id: str):
        """Verify a screening result from history."""
        workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID)
        if not workflow.load_results(self.results_file):
            print("No results file found. Run a screening first.")
            return
        result = workflow.verify_result_from_history(job_id)
        print(json.dumps(result, indent=4))

    def run_history(self):
        """Display the history of screening jobs."""
        workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID)
        if not workflow.load_results(self.results_file):
            print("No results file found. Run a screening first.")
            return
        workflow.show_history()

if __name__ == "__main__":
    cli = PureProtCLI()
    cli.run()




