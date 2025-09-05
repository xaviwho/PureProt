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
from modeling.docking_engine import DockingEngine, HybridScreening
from modeling.molecular_modeling import ScreeningPipeline

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

        # --- Convert Command ---
        convert_parser = self.subparsers.add_parser("convert", help="Convert a .smi file to a PureProt-compatible .csv file.")
        convert_parser.add_argument("input_path", type=str, help="Path to the input .smi file.")
        convert_parser.add_argument("output_path", type=str, help="Path for the output .csv file.")

        # --- Prep Protein Command ---
        prep_parser = self.subparsers.add_parser("prep-protein", help="Prepare protein structure for docking.")
        prep_parser.add_argument("pdb_path", type=str, help="Path to input PDB file.")
        prep_parser.add_argument("--output", type=str, help="Path for output PDBQT file.")

        # --- Dock Command ---
        dock_parser = self.subparsers.add_parser("dock", help="Perform molecular docking on molecules.")
        dock_parser.add_argument("csv_path", type=str, help="Path to CSV file containing molecules to dock.")
        dock_parser.add_argument("--protein", type=str, required=True, help="Path to prepared protein PDBQT file.")
        dock_parser.add_argument("--center", type=str, required=True, help="Binding site center coordinates (x,y,z).")
        dock_parser.add_argument("--size", type=str, default="20,20,20", help="Search box size (x,y,z). Default: 20,20,20")

        # --- Hybrid Screen Command ---
        hybrid_parser = self.subparsers.add_parser("hybrid-screen", help="Perform hybrid AI+docking screening.")
        hybrid_parser.add_argument("csv_path", type=str, help="Path to CSV file containing molecules to screen.")
        hybrid_parser.add_argument("--model", type=str, help="Path to custom AI model file.")
        hybrid_parser.add_argument("--protein", type=str, help="Path to prepared protein PDBQT file for docking.")
        hybrid_parser.add_argument("--center", type=str, help="Binding site center coordinates (x,y,z).")
        hybrid_parser.add_argument("--size", type=str, default="20,20,20", help="Search box size (x,y,z). Default: 20,20,20")

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
        elif args.command == "convert":
            self.run_convert(args.input_path, args.output_path)
        elif args.command == "prep-protein":
            self.run_prep_protein(args.pdb_path, args.output)
        elif args.command == "dock":
            self.run_dock(args.csv_path, args.protein, args.center, args.size)
        elif args.command == "hybrid-screen":
            self.run_hybrid_screen(args.csv_path, args.model, args.protein, args.center, args.size)
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
        convert     : Convert a .smi file into a PureProt-compatible .csv file.
        prep-protein: Prepare a protein structure for molecular docking.
        dock        : Perform molecular docking on a batch of molecules.
        hybrid-screen: Perform hybrid AI+docking screening with consensus scoring.

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

    def run_convert(self, smi_path: str, csv_path: str):
        """Converts a .smi file to a .csv file compatible with PureProt."""
        print(f"--- Converting SMI file: {smi_path} ---")
        try:
            with open(smi_path, 'r') as f_in, open(csv_path, 'w', newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerow(['molecule_id', 'smiles'])
                
                count = 0
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        smiles = parts[0].strip()
                        molecule_id = ' '.join(parts[1:]).strip()
                        writer.writerow([molecule_id, smiles])
                        count += 1
                    else:
                        print(f"Warning: Skipping malformed line: {line}")
                
                print(f"\nSuccessfully converted {count} molecules.")
                print(f"Output saved to: {csv_path}")

        except FileNotFoundError:
            print(f"Error: The file was not found at {smi_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        print("\n--- Conversion Complete ---")

    def run_prep_protein(self, pdb_path: str, output_path: Optional[str] = None):
        """Prepare protein structure for molecular docking."""
        print(f"--- Preparing Protein: {pdb_path} ---")
        
        if not output_path:
            output_path = pdb_path.replace('.pdb', '_prepared.pdbqt')
        
        try:
            docking_engine = DockingEngine()
            prepared_path = docking_engine.prepare_protein(pdb_path, output_path)
            print(f"Protein prepared successfully: {prepared_path}")
        except Exception as e:
            print(f"Error preparing protein: {e}")

    def run_dock(self, csv_path: str, protein_path: str, center_str: str, size_str: str = "20,20,20"):
        """Perform molecular docking on a batch of molecules."""
        print(f"--- Molecular Docking ---")
        print(f"Molecules: {csv_path}")
        print(f"Protein: {protein_path}")
        
        try:
            # Parse coordinates
            center = tuple(map(float, center_str.split(',')))
            size = tuple(map(float, size_str.split(',')))
            
            # Initialize docking engine
            docking_engine = DockingEngine(protein_path)
            docking_engine.set_binding_site(center, size)
            
            # Load molecules
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                molecules = list(reader)
            
            print(f"Docking {len(molecules)} molecules...")
            
            # Perform docking
            results = docking_engine.batch_dock(molecules)
            
            # Save results
            output_path = csv_path.replace('.csv', '_docking_results.csv')
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            
            print(f"\nDocking complete. Results saved to: {output_path}")
            
            # Show summary
            successful = [r for r in results if r.get('status') == 'success']
            print(f"Successfully docked: {len(successful)}/{len(results)} molecules")
            
            if successful:
                scores = [r['docking_score'] for r in successful]
                print(f"Best docking score: {min(scores):.2f}")
                print(f"Average docking score: {sum(scores)/len(scores):.2f}")
                
        except Exception as e:
            print(f"Error during docking: {e}")

    def run_hybrid_screen(self, csv_path: str, model_path: Optional[str] = None, 
                         protein_path: Optional[str] = None, center_str: Optional[str] = None, 
                         size_str: str = "20,20,20"):
        """Perform hybrid AI+docking screening with consensus scoring."""
        print(f"--- Hybrid AI+Docking Screening ---")
        print(f"Molecules: {csv_path}")
        
        try:
            # Initialize AI pipeline
            ai_pipeline = None
            if model_path:
                ai_pipeline = ScreeningPipeline(model_path)
                print(f"AI model loaded: {model_path}")
            
            # Initialize docking engine
            docking_engine = None
            if protein_path and center_str:
                center = tuple(map(float, center_str.split(',')))
                size = tuple(map(float, size_str.split(',')))
                docking_engine = DockingEngine(protein_path)
                docking_engine.set_binding_site(center, size)
                print(f"Docking configured: {protein_path}")
            
            if not ai_pipeline and not docking_engine:
                print("Error: At least one method (AI model or docking) must be specified")
                return
            
            # Initialize hybrid screening
            hybrid = HybridScreening(ai_pipeline, docking_engine)
            
            # Load molecules
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                molecules = list(reader)
            
            print(f"Screening {len(molecules)} molecules with hybrid approach...")
            
            # Perform hybrid screening
            results = []
            for i, mol in enumerate(molecules):
                mol_id = mol.get('molecule_id', f'mol_{i}')
                smiles = mol.get('smiles', '')
                
                print(f"Processing {mol_id} ({i+1}/{len(molecules)})")
                result = hybrid.hybrid_screen(mol_id, smiles)
                results.append(result)
            
            # Save results
            output_path = csv_path.replace('.csv', '_hybrid_results.csv')
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            
            print(f"\nHybrid screening complete. Results saved to: {output_path}")
            
            # Show summary
            consensus_scores = [r.get('consensus_score') for r in results if r.get('consensus_score') is not None]
            if consensus_scores:
                print(f"Consensus scores calculated for {len(consensus_scores)} molecules")
                print(f"Best consensus score: {max(consensus_scores):.3f}")
                print(f"Average consensus score: {sum(consensus_scores)/len(consensus_scores):.3f}")
            
            # Record results on blockchain if available
            try:
                workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID)
                workflow.load_results(self.results_file)
                
                for result in results:
                    if result.get('consensus_score') is not None:
                        # Create blockchain record for hybrid result
                        blockchain_data = {
                            'screening_type': 'hybrid',
                            'ai_prediction': result.get('predicted_pIC50'),
                            'docking_score': result.get('docking_score'),
                            'consensus_score': result.get('consensus_score'),
                            'drug_like': result.get('drug_like', False)
                        }
                        
                        # Record on blockchain (simplified)
                        mol_id = result['molecule_id']
                        smiles = result['smiles']
                        workflow.run_screening_job(mol_id, smiles, custom_data=blockchain_data)
                
                workflow.save_results(self.results_file)
                print("Results recorded on blockchain for verification")
                
            except Exception as e:
                print(f"Warning: Could not record on blockchain: {e}")
                
        except Exception as e:
            print(f"Error during hybrid screening: {e}")


if __name__ == "__main__":
    cli = PureProtCLI()
    cli.run()

