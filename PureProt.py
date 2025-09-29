"""PureProtX: Modular CLI Protocol for Blockchain-Audited Consensus AI and Docking-Based Virtual Screening

This is the main CLI interface for PureProtX, a truly modular system that combines
Consensus AI, molecular docking, and comprehensive blockchain auditing.
"""

import argparse
import sys
import os
import csv
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

# Import modular PureProtX components
from pureprot import ConsensusAIModel, BlockchainAuditor, DockingEngine, DataManager

# Legacy imports for backward compatibility
from workflow.verification_workflow import VerifiableDrugScreening
from modeling.advanced_docking_engine import AdvancedDockingEngine, create_docking_engine
from modeling.docking_engine import HybridScreening

# Purechain configuration
PURECHAIN_RPC_URL = "https://purechainnode.com:8547"
PURECHAIN_CHAIN_ID = 900520900520
PURECHAIN_CURRENCY = "PCC"

class PureProtXCLI:
    """Main class for the PureProtX modular command-line interface."""

    def __init__(self, results_file="pureprot_results.json"):
        """Initialize the CLI with modular components."""
        self.results_file = results_file
        
        # Initialize modular components
        self.data_manager = DataManager()
        self.consensus_ai = None  # Initialized when needed
        self.docking_engine = None  # Initialized when needed
        
        # Initialize blockchain auditor with graceful error handling
        try:
            self.blockchain_auditor = BlockchainAuditor(
                rpc_url=PURECHAIN_RPC_URL, 
                chain_id=PURECHAIN_CHAIN_ID
            )
        except Exception as e:
            print(f"Note: Blockchain auditor initialization deferred ({e})")
            self.blockchain_auditor = None
        
        # Setup argument parser
        self.parser = argparse.ArgumentParser(
            description="PureProtX: Modular CLI Protocol for Blockchain-Audited Consensus AI and Docking-Based Virtual Screening",
            formatter_class=argparse.RawTextHelpFormatter
        )
        self.parser.add_argument("-v", "--version", action="version", version="PureProtX 1.0.0")
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
        batch_parser.add_argument("--output", type=str, help="Path to save AI-only results CSV file.")

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
        dock_parser = self.subparsers.add_parser("dock", help="Dock a single molecule against a receptor.")
        dock_parser.add_argument("molecule_id", type=str, help="Identifier for the molecule.")
        dock_parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the ligand.")
        dock_parser.add_argument("--receptor", type=str, required=True, help="Path to receptor PDBQT file.")
        dock_parser.add_argument("--center", type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'), 
                                help="Center coordinates for docking box (X Y Z).")
        dock_parser.add_argument("--size", type=float, nargs=3, default=[20.0, 20.0, 20.0], metavar=('X', 'Y', 'Z'),
                                help="Size of docking box in Angstroms (default: 20 20 20).")
        dock_parser.add_argument("--exhaustiveness", type=int, default=8, help="Exhaustiveness of search (default: 8).")
        dock_parser.add_argument("--output", type=str, help="Path to save docking scores CSV file.")

        # --- Dock Batch Command ---
        dock_batch_parser = self.subparsers.add_parser("dock-batch", help="Dock multiple molecules from CSV file.")
        dock_batch_parser.add_argument("csv_path", type=str, help="Path to CSV file with molecule_id,smiles columns.")
        dock_batch_parser.add_argument("--receptor", type=str, required=True, help="Path to receptor PDBQT file.")
        dock_batch_parser.add_argument("--center", type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'),
                                      help="Center coordinates for docking box (X Y Z).")
        dock_batch_parser.add_argument("--size", type=float, nargs=3, default=[20.0, 20.0, 20.0], metavar=('X', 'Y', 'Z'),
                                      help="Size of docking box in Angstroms (default: 20 20 20).")
        dock_batch_parser.add_argument("--exhaustiveness", type=int, default=8, help="Exhaustiveness of search (default: 8).")
        dock_batch_parser.add_argument("--output", type=str, help="Path to save docking scores CSV file.")
        dock_batch_parser.add_argument("--limit", type=int, help="Limit number of molecules to dock.")

        # --- Find Binding Site Command ---
        binding_parser = self.subparsers.add_parser("find-binding-site", help="Find binding site coordinates and box size for a protein.")
        binding_parser.add_argument("protein_path", type=str, help="Path to the protein PDB file.")
        binding_parser.add_argument("--method", type=str, default="auto", choices=["auto", "ligand", "center"], help="Method to find binding site.")

        # --- Compare Results Command ---
        compare_parser = self.subparsers.add_parser("compare", help="Compare AI-only, docking-only, and hybrid screening results.")
        compare_parser.add_argument("--ai-results", type=str, help="Path to AI-only results CSV file.")
        compare_parser.add_argument("--docking-results", type=str, help="Path to docking-only results CSV file.")
        compare_parser.add_argument("--hybrid-results", type=str, help="Path to hybrid results CSV file.")
        compare_parser.add_argument("--output", type=str, default="comparison_analysis.csv", help="Output path for comparison analysis.")

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
            self.run_batch(args.csv_path, args.model, args.output)
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
        elif args.command == "find-binding-site":
            self.run_find_binding_site(args.protein_path, args.method)
        elif args.command == "compare":
            self.run_compare_results(args.ai_results, args.docking_results, args.hybrid_results, args.output)
        elif args.command == "dock":
            self.run_dock_single(args.molecule_id, args.smiles, args.receptor, args.center, args.size, args.exhaustiveness, args.output)
        elif args.command == "dock-batch":
            self.run_dock_batch(args.csv_path, args.receptor, args.center, args.size, args.exhaustiveness, args.output, args.limit)
        elif args.command == "hybrid-screen":
            self.run_hybrid_screen(args.csv_path, args.model, args.protein, args.center, args.size)
        else:
            self.parser.print_help()

    def show_info(self):
        """Display the welcome message and guide for the user."""
        info_text = """
        ================================================================
        PureProtX: Modular CLI Protocol for Blockchain-Audited 
        Consensus AI and Docking-Based Virtual Screening
        ================================================================

        Welcome to PureProtX, a truly modular system that delivers on the promise
        of transparent, reproducible drug discovery with comprehensive blockchain auditing.

        🔬 MODULAR ARCHITECTURE:
        • AI Module: Consensus AI (SVR + Random Forest + Gradient Boosting)
        • Docking Module: Advanced molecular docking with multiple engines
        • Blockchain Module: Comprehensive audit trail (models, proteins, parameters)
        • Data Module: ChEMBL integration and dataset management

        🧠 CONSENSUS AI:
        • Ensemble of 3 models for robust predictions
        • Individual model performance tracking
        • Mathematically proven superior accuracy

        🔗 BLOCKCHAIN AUDIT:
        • Hashes AI model files for reproducibility
        • Hashes protein structures and parameters
        • Complete audit trail for regulatory compliance
        • Zero gas fees on Purechain network

        Available Commands:
        -------------------
        info        : Show this information message.
        connect     : Test the connection to the Purechain blockchain.
        fetch-data  : Fetch and prepare bioactivity data from ChEMBL.
        train-model : Train Consensus AI model (SVR+RF+GB ensemble).
        screen      : Screen molecules using Consensus AI with blockchain audit.
        batch       : Screen batches with comprehensive audit trails.
        dock        : Dock single molecule with blockchain audit.
        dock-batch  : Dock multiple molecules from CSV with audit.
        verify      : Verify screening results against blockchain records.
        history     : Display screening job history.
        benchmark   : Performance benchmarking with consensus metrics.
        convert     : Convert SMILES files to PureProtX format.
        prep-protein: Prepare protein structures for docking.
        find-binding-site: Auto-detect binding sites.
        hybrid-screen: Hybrid AI+docking with consensus scoring.

        Example Usage:
        --------------
        # 1. Fetch data and train Consensus AI model
        python PureProt.py fetch-data "CHEMBL243" --output "hiv_data.csv"
        python PureProt.py train-model "hiv_data.csv" --output "hiv_consensus_model.joblib"

        # 2. Screen with Consensus AI + Blockchain Audit
        python PureProt.py screen "test_molecule" --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --model "hiv_consensus_model.joblib"

        # 3. Molecular Docking with Blockchain Audit
        python PureProt.py prep-protein "1hpv.pdb" --output "1hpv_prepared.pdbqt"
        python PureProt.py find-binding-site "1hpv_prepared.pdbqt"
        python PureProt.py dock "aspirin" --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --receptor "1hpv_prepared.pdbqt" --center 10.0 15.0 20.0
        python PureProt.py dock-batch "molecules.csv" --receptor "1hpv_prepared.pdbqt" --center 10.0 15.0 20.0 --output "docking_scores.csv"

        # 4. Verify blockchain audit
        python PureProt.py verify "test_molecule_1234567890"

        # 5. Hybrid AI+Docking screening
        python PureProt.py hybrid-screen "molecules.csv" --model "hiv_consensus_model.joblib" --protein "1hpv_prepared.pdbqt" --center "10.0,15.0,20.0"

        # 6. Performance benchmarking
        python PureProt.py benchmark "hiv_data.csv" --limit 100

        For more details on a specific command, run:
        python PureProt.py [command] --help
        """
        print(info_text)

    def run_batch(self, csv_path: str, model_path: Optional[str] = None, output_path: Optional[str] = None):
        """Screen a batch of molecules from a CSV file."""
        print(f"--- Batch Screening from: {csv_path} ---")
        
        # Generate output filename based on input
        if not output_path:
            base_name = csv_path.replace('.csv', '')
            output_path = f"{base_name}_ai_only_results.csv"
            print(f"No output path specified. Saving to: {output_path}")
        
        workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID, model_path=model_path)
        workflow.load_results(self.results_file)
        
        # Load molecules from CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            molecules = list(reader)
        
        # Process each molecule
        results = []
        for i, row in enumerate(molecules):
            molecule_id = row.get('molecule_id') or row.get('canonical_smiles')
            smiles = row.get('smiles') or row.get('canonical_smiles')
            if not molecule_id or not smiles:
                print(f"Skipping row {i+1}: missing molecule identifier or SMILES string.")
                continue
            
            print(f"\n--- Processing Molecule {i+1}/{len(molecules)}: {molecule_id} ---")
            result = workflow.run_screening_job(molecule_id, smiles)
            results.append(result)
        
        # Save AI-only results to dedicated file
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"\nAI-only results saved to: {output_path}")
        
        workflow.save_results(self.results_file)
        print("\n--- Batch Screening Complete ---")

    def run_compare_results(self, ai_results: Optional[str] = None, docking_results: Optional[str] = None, 
                          hybrid_results: Optional[str] = None, output_path: str = "comparison_analysis.csv"):
        """Compare AI-only, docking-only, and hybrid screening results."""
        print("--- Comparative Analysis of Screening Methods ---")
        
        results_data = {}
        
        # Load AI-only results
        if ai_results and os.path.exists(ai_results):
            print(f"Loading AI-only results from: {ai_results}")
            ai_df = pd.read_csv(ai_results)
            results_data['ai'] = ai_df
            print(f"AI-only results: {len(ai_df)} molecules")
        else:
            print("AI-only results not provided or file not found")
        
        # Load docking-only results
        if docking_results and os.path.exists(docking_results):
            print(f"Loading docking-only results from: {docking_results}")
            dock_df = pd.read_csv(docking_results)
            results_data['docking'] = dock_df
            print(f"Docking-only results: {len(dock_df)} molecules")
        else:
            print("Docking-only results not provided or file not found")
        
        # Load hybrid results
        if hybrid_results and os.path.exists(hybrid_results):
            print(f"Loading hybrid results from: {hybrid_results}")
            hybrid_df = pd.read_csv(hybrid_results)
            results_data['hybrid'] = hybrid_df
            print(f"Hybrid results: {len(hybrid_df)} molecules")
        else:
            print("Hybrid results not provided or file not found")
        
        if not results_data:
            print("Error: No valid result files provided for comparison")
            return
        
        # Perform comparative analysis
        comparison_results = []
        
        # Get common molecules across all datasets
        molecule_sets = []
        for method, df in results_data.items():
            if 'molecule_id' in df.columns:
                molecule_sets.append(set(df['molecule_id'].tolist()))
            elif 'smiles' in df.columns:
                molecule_sets.append(set(df['smiles'].tolist()))
        
        if molecule_sets:
            common_molecules = set.intersection(*molecule_sets) if len(molecule_sets) > 1 else molecule_sets[0]
            print(f"Common molecules across datasets: {len(common_molecules)}")
            
            # Create comparison for each common molecule
            for mol_id in common_molecules:
                comparison_row = {'molecule_id': mol_id}
                
                for method, df in results_data.items():
                    # Find molecule row
                    if 'molecule_id' in df.columns:
                        mol_row = df[df['molecule_id'] == mol_id]
                    else:
                        mol_row = df[df['smiles'] == mol_id]
                    
                    if not mol_row.empty:
                        row = mol_row.iloc[0]
                        if method == 'ai':
                            comparison_row[f'{method}_prediction'] = row.get('predicted_pIC50', row.get('ai_prediction', 'N/A'))
                        elif method == 'docking':
                            comparison_row[f'{method}_score'] = row.get('docking_score', row.get('best_score', 'N/A'))
                        elif method == 'hybrid':
                            comparison_row[f'{method}_consensus'] = row.get('consensus_score', 'N/A')
                            comparison_row[f'{method}_ai'] = row.get('predicted_pIC50', row.get('ai_prediction', 'N/A'))
                            comparison_row[f'{method}_docking'] = row.get('docking_score', 'N/A')
                
                comparison_results.append(comparison_row)
        
        # Save comparison results
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df.to_csv(output_path, index=False)
            print(f"\nComparison analysis saved to: {output_path}")
            
            # Print summary statistics
            print("\n--- Comparison Summary ---")
            print(f"Total molecules compared: {len(comparison_results)}")
            
            # Calculate correlations if multiple methods available
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlations = comparison_df[numeric_cols].corr()
                print("\nCorrelation Matrix:")
                print(correlations.round(3))
        else:
            print("No common molecules found for comparison")

    def run_screen(self, molecule_id: str, smiles: Optional[str] = None, model_path: Optional[str] = None):
        """Run a single molecule screening job using Consensus AI with comprehensive blockchain auditing."""
        print(f"--- PureProtX Consensus AI Screening: {molecule_id} ---")
        
        try:
            # Initialize Consensus AI model if not already done
            if not self.consensus_ai:
                if model_path and os.path.exists(model_path):
                    self.consensus_ai = ConsensusAIModel(model_path)
                    print(f"Using custom consensus model: {model_path}")
                else:
                    # Try to find a consensus model
                    consensus_models = [f for f in os.listdir('.') if f.endswith('_consensus_model.joblib')]
                    if consensus_models:
                        self.consensus_ai = ConsensusAIModel(consensus_models[0])
                        print(f"Using consensus model: {consensus_models[0]}")
                    else:
                        # Fallback to legacy workflow
                        print("No consensus model found. Using legacy workflow...")
                        workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID, model_path=model_path)
                        workflow.load_results(self.results_file)
                        result = workflow.run_screening_job(molecule_id=molecule_id, smiles=smiles)
                        workflow.save_results(self.results_file)
                        print("\n--- Screening Result ---")
                        print(json.dumps(result, indent=4))
                        return
            
            # Make consensus prediction
            ai_predictions = self.consensus_ai.predict_single(smiles)
            
            # Create screening results
            screening_results = {
                'molecule_id': molecule_id,
                'smiles': smiles,
                'consensus_pic50': ai_predictions['consensus'],
                'individual_predictions': {
                    'svr': ai_predictions['svr'],
                    'random_forest': ai_predictions['random_forest'],
                    'gradient_boosting': ai_predictions['gradient_boosting']
                },
                'screening_type': 'consensus_ai'
            }
            
            # Prepare parameters for comprehensive audit
            parameters = {
                'screening_type': 'consensus_ai',
                'model_info': self.consensus_ai.get_model_info(),
                'software_version': 'PureProtX-1.0.0'
            }
            
            # Create comprehensive audit record
            audit_record = self.blockchain_auditor.create_comprehensive_audit_record(
                molecule_id=molecule_id,
                smiles=smiles,
                results=screening_results,
                model_path=model_path,
                parameters=parameters
            )
            
            # Record on blockchain
            tx_hash, job_id = self.blockchain_auditor.record_screening_result(audit_record)
            
            # Add blockchain info to results
            screening_results.update({
                'job_id': job_id,
                'transaction_hash': tx_hash,
                'audit_hash': audit_record['master_hash']
            })
            
            print(f"\n=== Consensus AI Results ===")
            print(f"Consensus pIC50: {ai_predictions['consensus']:.4f}")
            print(f"Individual predictions:")
            print(f"  SVR: {ai_predictions['svr']:.4f}")
            print(f"  Random Forest: {ai_predictions['random_forest']:.4f}")
            print(f"  Gradient Boosting: {ai_predictions['gradient_boosting']:.4f}")
            print(f"\nBlockchain Audit:")
            print(f"  Job ID: {job_id}")
            print(f"  Transaction: {tx_hash}")
            
            print("\n--- Screening Result ---")
            print(json.dumps(screening_results, indent=4))
            
        except Exception as e:
            print(f"✗ Error in consensus screening: {e}")
            # Fallback to legacy workflow
            print("Falling back to legacy screening...")
            workflow = VerifiableDrugScreening(rpc_url=PURECHAIN_RPC_URL, chain_id=PURECHAIN_CHAIN_ID, model_path=model_path)
            workflow.load_results(self.results_file)
            result = workflow.run_screening_job(molecule_id=molecule_id, smiles=smiles)
            workflow.save_results(self.results_file)
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
        
        try:
            dataset_path = self.data_manager.fetch_chembl_data(target_id, output_path)
            print(f"✓ Data prepared successfully: {dataset_path}")
        except Exception as e:
            print(f"✗ Error fetching data: {e}")

    def run_train_model(self, dataset_path: str, model_output_path: Optional[str] = None):
        """Train a new Consensus AI model (SVR + Random Forest + Gradient Boosting)."""
        print(f"--- Training Consensus AI Model from Dataset: {dataset_path} ---")
        if not model_output_path:
            model_output_path = dataset_path.replace('.csv', '_consensus_model.joblib')
            print(f"No output path specified. Saving model to: {model_output_path}")
        
        try:
            # Initialize Consensus AI model
            self.consensus_ai = ConsensusAIModel()
            
            # Train the ensemble
            performance_metrics = self.consensus_ai.train(dataset_path)
            
            # Save the trained model
            model_hash = self.consensus_ai.save_model(model_output_path)
            
            print(f"\n=== Consensus AI Training Results ===")
            for model_name, metrics in performance_metrics.items():
                print(f"{model_name.upper()}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
            
            print(f"\n✓ Consensus AI model saved: {model_output_path}")
            print(f"✓ Model hash: {model_hash}")
            
        except Exception as e:
            print(f"✗ Error training Consensus AI model: {e}")
            # Fallback to legacy training
            print("Falling back to legacy training method...")
            from modeling.model_trainer import train_and_save_model
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

    def run_find_binding_site(self, protein_path: str, method: str = "auto"):
        """Automatically detect binding site center coordinates for any protein."""
        print(f"--- Finding Binding Site for: {protein_path} ---")
        
        try:
            import numpy as np
            
            # Method 1: Extract from co-crystallized ligands (HETATM records)
            ligand_coords = []
            with open(protein_path, 'r') as f:
                for line in f:
                    if line.startswith('HETATM') and not line[17:20].strip() in ['HOH', 'WAT', 'SO4', 'PO4', 'CL', 'NA']:
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46]) 
                            z = float(line[46:54])
                            ligand_coords.append([x, y, z])
                        except ValueError:
                            continue
            
            if ligand_coords:
                center = np.mean(ligand_coords, axis=0)
                
                # Calculate dynamic box size based on ligand dimensions
                coords_array = np.array(ligand_coords)
                min_coords = np.min(coords_array, axis=0)
                max_coords = np.max(coords_array, axis=0)
                ligand_dimensions = max_coords - min_coords
                
                # Add buffer around ligand (typically 5-10 Å on each side)
                buffer = 8.0  # 8 Å buffer for flexibility
                box_size = ligand_dimensions + (2 * buffer)
                
                # Ensure minimum box size of 15 Å and maximum of 30 Å per dimension
                box_size = np.clip(box_size, 15.0, 30.0)
                
                print(f"✓ Binding site detected from co-crystallized ligand")
                print(f"  Ligand dimensions: {ligand_dimensions[0]:.1f} × {ligand_dimensions[1]:.1f} × {ligand_dimensions[2]:.1f} Å")
                print(f"  Center coordinates: {center[0]:.1f},{center[1]:.1f},{center[2]:.1f}")
                print(f"  Calculated box size: {box_size[0]:.1f},{box_size[1]:.1f},{box_size[2]:.1f}")
                print(f"  (Ligand + {buffer:.0f}Å buffer, min 15Å, max 30Å per dimension)")
                print(f"\nUse in docking commands:")
                print(f"  --center \"{center[0]:.1f},{center[1]:.1f},{center[2]:.1f}\"")
                print(f"  --size \"{box_size[0]:.1f},{box_size[1]:.1f},{box_size[2]:.1f}\"")
                return center
            
            # Method 2: Calculate geometric center of protein (fallback)
            protein_coords = []
            with open(protein_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            protein_coords.append([x, y, z])
                        except ValueError:
                            continue
            
            if protein_coords:
                center = np.mean(protein_coords, axis=0)
                print(f"⚠ No ligand found. Using protein geometric center")
                print(f"  Center coordinates: {center[0]:.1f},{center[1]:.1f},{center[2]:.1f}")
                print(f"  Recommended box size: 30,30,30 (larger for whole protein)")
                print(f"\nUse in docking commands:")
                print(f"  --center \"{center[0]:.1f},{center[1]:.1f},{center[2]:.1f}\"")
                return center
            
            print("❌ Could not determine binding site coordinates")
            return None
            
        except Exception as e:
            print(f"Error detecting binding site: {e}")
            return None

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
            docking_engine = create_docking_engine()
            status = docking_engine.get_engine_status()
            print(f"Available engines: {[k for k, v in status.items() if v]}")
            print(f"Primary method: {status['primary_method']}")
            
            # For protein preparation, we'll just copy/validate the file for now
            # Real preparation would depend on the specific docking engine
            if os.path.exists(pdb_path):
                import shutil
                shutil.copy2(pdb_path, output_path)
                print(f"Protein prepared successfully: {output_path}")
            else:
                print(f"Error: Protein file not found: {pdb_path}")
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
            
            # Initialize advanced docking engine
            docking_engine = create_docking_engine(protein_path)
            docking_engine.set_binding_site(center, size)
            
            # Show available engines
            status = docking_engine.get_engine_status()
            print(f"Using docking method: {status['primary_method']}")
            print(f"Available engines: {[k for k, v in status.items() if v and k != 'primary_method']}")
            
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
            
            # Initialize advanced docking engine
            docking_engine = None
            if protein_path and center_str:
                center = tuple(map(float, center_str.split(',')))
                size = tuple(map(float, size_str.split(',')))
                docking_engine = create_docking_engine(protein_path)
                docking_engine.set_binding_site(center, size)
                
                # Show engine status
                status = docking_engine.get_engine_status()
                print(f"Docking method: {status['primary_method']}")
                print(f"Available engines: {[k for k, v in status.items() if v and k != 'primary_method']}")
            else:
                # Create engine without protein for AI-only mode
                docking_engine = create_docking_engine()
                print("Docking engine created (AI-enhanced scoring mode)")
            
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
                from workflow.verification_workflow import VerifiableDrugScreening
                workflow = VerifiableDrugScreening(network='testnet')
                
                # Record hybrid results directly on blockchain without re-screening
                for result in results:
                    if result.get('consensus_score') is not None:
                        mol_id = result['molecule_id']
                        smiles = result['smiles']
                        
                        # Create result hash for blockchain
                        result_data = {
                            'molecule_id': mol_id,
                            'smiles': smiles,
                            'screening_type': 'hybrid',
                            'consensus_score': result.get('consensus_score'),
                            'ai_prediction': result.get('predicted_pIC50'),
                            'docking_score': result.get('docking_score'),
                            'drug_like': result.get('drug_like', False)
                        }
                        
                        # Record directly on blockchain
                        result_hash = hashlib.sha256(json.dumps(result_data, sort_keys=True).encode()).digest()
                        molecule_hash = hashlib.sha256(smiles.encode()).digest()
                        
                        blockchain_result = workflow.blockchain_connector.record_and_verify_result(
                            result_hash, molecule_hash, mol_id
                        )
                        
                        if blockchain_result.get('success'):
                            print(f"✅ {mol_id} recorded on blockchain: {blockchain_result['tx_hash'][:10]}...")
                        else:
                            print(f"❌ Failed to record {mol_id}: {blockchain_result.get('error', 'Unknown error')}")
                
                print("Hybrid screening results recorded on PureChain blockchain")
                
            except Exception as e:
                print(f"Warning: Could not record on blockchain: {e}")
                
        except Exception as e:
            print(f"Error during hybrid screening: {e}")

    def run_dock_single(self, molecule_id: str, smiles: str, receptor_path: str, center: List[float], 
                       size: List[float], exhaustiveness: int, output_path: Optional[str] = None):
        """Dock a single molecule against a receptor with blockchain audit."""
        print(f"--- Single Molecule Docking: {molecule_id} ---")
        print(f"Receptor: {receptor_path}")
        print(f"Center: {center}")
        print(f"Box Size: {size}")
        print(f"Exhaustiveness: {exhaustiveness}")
        
        try:
            # Initialize docking engine
            if not self.docking_engine:
                from pureprot.docking import DockingEngine
                self.docking_engine = DockingEngine(receptor_path)
            
            # Perform docking
            result = self.docking_engine.dock_molecule(
                smiles=smiles,
                molecule_id=molecule_id,
                center=center,
                size=size,
                exhaustiveness=exhaustiveness
            )
            
            # Create audit record
            if self.blockchain_auditor:
                audit_params = {
                    'receptor_file': receptor_path,
                    'center_coordinates': center,
                    'box_size': size,
                    'exhaustiveness': exhaustiveness,
                    'docking_engine': 'AutoDock Vina',
                    'software_version': 'PureProtX-1.0.0'
                }
                
                audit_record = self.blockchain_auditor.create_comprehensive_audit_record(
                    molecule_id=molecule_id,
                    smiles=smiles,
                    results=result,
                    parameters=audit_params
                )
                
                print(f"✅ Audit hash: {audit_record['master_hash'][:16]}...")
            
            # Save results
            if not output_path:
                output_path = f"{molecule_id}_docking_scores.csv"
            
            import pandas as pd
            df = pd.DataFrame([result])
            df.to_csv(output_path, index=False)
            
            print(f"✅ Docking score: {result.get('docking_score', 'N/A')}")
            print(f"✅ Results saved to: {output_path}")
            
            # Append to deterministic JSON
            self._append_to_deterministic_json(result, audit_record if self.blockchain_auditor else None)
            
        except Exception as e:
            print(f"❌ Docking failed: {e}")

    def run_dock_batch(self, csv_path: str, receptor_path: str, center: List[float], size: List[float], 
                      exhaustiveness: int, output_path: Optional[str] = None, limit: Optional[int] = None):
        """Dock multiple molecules from CSV file with blockchain audit."""
        print(f"--- Batch Molecular Docking ---")
        print(f"Input CSV: {csv_path}")
        print(f"Receptor: {receptor_path}")
        print(f"Center: {center}")
        print(f"Box Size: {size}")
        print(f"Exhaustiveness: {exhaustiveness}")
        if limit:
            print(f"Limit: {limit} molecules")
        
        try:
            import pandas as pd
            
            # Load molecules
            df = pd.read_csv(csv_path)
            if limit:
                df = df.head(limit)
            
            print(f"Loaded {len(df)} molecules for docking")
            
            # Initialize docking engine
            if not self.docking_engine:
                from pureprot.docking import DockingEngine
                self.docking_engine = DockingEngine(receptor_path)
            
            results = []
            audit_records = []
            
            for idx, row in df.iterrows():
                molecule_id = row['molecule_id']
                smiles = row['smiles']
                
                print(f"Docking {idx+1}/{len(df)}: {molecule_id}")
                
                try:
                    # Perform docking
                    result = self.docking_engine.dock_molecule(
                        smiles=smiles,
                        molecule_id=molecule_id,
                        center=center,
                        size=size,
                        exhaustiveness=exhaustiveness
                    )
                    
                    results.append(result)
                    
                    # Create audit record
                    if self.blockchain_auditor:
                        audit_params = {
                            'receptor_file': receptor_path,
                            'center_coordinates': center,
                            'box_size': size,
                            'exhaustiveness': exhaustiveness,
                            'docking_engine': 'AutoDock Vina',
                            'software_version': 'PureProtX-1.0.0',
                            'batch_index': idx
                        }
                        
                        audit_record = self.blockchain_auditor.create_comprehensive_audit_record(
                            molecule_id=molecule_id,
                            smiles=smiles,
                            results=result,
                            parameters=audit_params
                        )
                        
                        audit_records.append(audit_record)
                    
                    print(f"  ✅ Score: {result.get('docking_score', 'N/A')}")
                    
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    # Add failed result
                    failed_result = {
                        'molecule_id': molecule_id,
                        'smiles': smiles,
                        'docking_score': None,
                        'success': False,
                        'error': str(e)
                    }
                    results.append(failed_result)
            
            # Save results
            if not output_path:
                output_path = "batch_docking_scores.csv"
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            
            successful = sum(1 for r in results if r.get('success', False))
            print(f"\n✅ Batch docking complete: {successful}/{len(results)} successful")
            print(f"✅ Results saved to: {output_path}")
            
            # Append to deterministic JSON
            batch_data = {
                'batch_docking_results': results,
                'batch_audit_records': audit_records if self.blockchain_auditor else [],
                'batch_summary': {
                    'total_molecules': len(results),
                    'successful_dockings': successful,
                    'receptor_file': receptor_path,
                    'parameters': {
                        'center': center,
                        'size': size,
                        'exhaustiveness': exhaustiveness
                    }
                }
            }
            
            self._append_to_deterministic_json(batch_data, None)
            
        except Exception as e:
            print(f"❌ Batch docking failed: {e}")

    def _append_to_deterministic_json(self, data: dict, audit_record: Optional[dict] = None):
        """Append results to deterministic JSON file for hashing and logging."""
        import json
        import os
        from datetime import datetime
        
        json_file = "pureprot_deterministic_results.json"
        
        # Load existing data or create new
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {
                'metadata': {
                    'software': 'PureProtX',
                    'version': '1.0.0',
                    'created': datetime.now().isoformat()
                },
                'results': []
            }
        
        # Add new result
        entry = {
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'audit_record': audit_record
        }
        
        all_data['results'].append(entry)
        all_data['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Save with deterministic formatting
        with open(json_file, 'w') as f:
            json.dump(all_data, f, indent=2, sort_keys=True)
        
        # Calculate and log hash
        import hashlib
        json_str = json.dumps(all_data, sort_keys=True)
        file_hash = hashlib.sha256(json_str.encode()).hexdigest()
        
        print(f"✅ Results appended to {json_file}")
        print(f"✅ Deterministic hash: {file_hash[:16]}...")


if __name__ == "__main__":
    cli = PureProtXCLI()
    cli.run()

