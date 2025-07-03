"""Main CLI Interface for Verifiable Drug Screening System

This module provides a command-line interface for interacting with the
verifiable drug screening system, including connecting to Purechain,
running screening jobs, and verifying results.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

from workflow.verification_workflow import VerifiableDrugScreening

# Purechain configuration
PURECHAIN_RPC_URL = "http://43.200.53.250:8548"
PURECHAIN_CHAIN_ID = 900520900520
PURECHAIN_CURRENCY = "PCC"


class DrugScreeningCLI:
    """Command-line interface for the drug screening system."""
    
    def __init__(self):
        """Initialize the CLI system."""
        self.workflow = VerifiableDrugScreening(PURECHAIN_RPC_URL)
        self.connected = False
        self.results_file = "screening_results.json"
    
    def connect(self) -> bool:
        """Connect to blockchain and initialize workflow.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        print(f"Connecting to Purechain at {PURECHAIN_RPC_URL}...")
        start_time = time.time()
        
        if self.workflow.connect_wallet():
            self.connected = True
            elapsed = time.time() - start_time
            print(f"✓ Connected to Purechain (Chain ID: {PURECHAIN_CHAIN_ID}, Currency: {PURECHAIN_CURRENCY})")
            print(f"  Connection established in {elapsed:.2f} seconds")
            return True
        else:
            print("✗ Failed to connect to blockchain")
            return False
    
    def screen(self, molecule_id: str, smiles: str, target_id: str = "default") -> Dict[str, Any]:
        """Run a screening job for a molecule.
        
        Args:
            molecule_id: Identifier for the molecule
            smiles: SMILES string representation of the molecule
            target_id: Identifier for the target protein
            
        Returns:
            Dict[str, Any]: Screening job result
        """
        if not self.connected and not self.connect():
            return {"error": "Not connected to blockchain"}
        
        print(f"Running screening for {molecule_id} against {target_id}...")
        start_time = time.time()
        
        result = self.workflow.run_screening_job(molecule_id, smiles, target_id)
        
        elapsed = time.time() - start_time
        print(f"Screening completed in {elapsed:.2f} seconds")
        
        # Print result summary
        if "error" not in result:
            print("\nScreening Results:")
            print(f"  Molecule: {result['molecule_id']}")
            print(f"  Target: {result['target_id']}")
            print(f"  Binding Affinity: {result['binding_affinity']} kcal/mol")
            print(f"  Toxicity Score: {result['toxicity_score']}")
            print(f"  Job ID: {result['job_id']}")
            
            if "verification" in result and "tx_hash" in result["verification"]:
                print(f"  Blockchain Verification: ✓")
                print(f"  Transaction Hash: {result['verification']['tx_hash']}")
            else:
                print(f"  Blockchain Verification: ✗")
        else:
            print(f"Error: {result['error']}")
        
        # Auto-save results
        self.workflow.save_results(self.results_file)
        
        return result
    
    def batch_screen(self, input_file: str) -> Dict[str, Dict[str, Any]]:
        """Run screening jobs for multiple molecules from a file.
        
        Args:
            input_file: Path to JSON file with molecule data
            
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of molecule IDs to results
        """
        if not self.connected and not self.connect():
            return {"error": "Not connected to blockchain"}
        
        try:
            # Load molecules from file
            with open(input_file, "r") as f:
                molecules = json.load(f)
            
            print(f"Loaded {len(molecules)} molecules from {input_file}")
            print("Starting batch screening...")
            start_time = time.time()
            
            results = self.workflow.batch_screen_molecules(molecules)
            
            elapsed = time.time() - start_time
            print(f"Batch screening completed in {elapsed:.2f} seconds")
            
            # Print summary
            print("\nBatch Screening Summary:")
            for mol_id, result in results.items():
                verification_status = "✓" if "verification" in result and "tx_hash" in result["verification"] else "✗"
                print(f"  {mol_id}: {result['binding_affinity']} kcal/mol, toxicity {result['toxicity_score']} [{verification_status}]")
            
            # Auto-save results
            self.workflow.save_results(self.results_file)
            
            return results
        except Exception as e:
            print(f"Error in batch screening: {e}")
            return {"error": str(e)}
    
    def verify(self, job_id: str, tx_hash: str) -> Dict[str, Any]:
        """Verify a previously recorded screening result.
        
        Args:
            job_id: The job ID to verify
            tx_hash: The blockchain transaction hash
            
        Returns:
            Dict[str, Any]: Verification results
        """
        if not self.connected and not self.connect():
            return {"error": "Not connected to blockchain"}
        
        print(f"Verifying job {job_id}...")
        start_time = time.time()
        
        verification = self.workflow.verify_result(job_id, tx_hash)
        
        elapsed = time.time() - start_time
        print(f"Verification completed in {elapsed:.2f} seconds")
        
        # Print verification result
        if "error" not in verification:
            status = "✓ Verified" if verification["verified"] else "✗ Failed"
            print("\nVerification Result:")
            print(f"  Status: {status}")
            print(f"  Job ID: {verification['job_id']}")
            print(f"  Molecule ID: {verification['molecule_id']}")
            print(f"  Time: {verification['verification_time']}")
        else:
            print(f"Error: {verification['error']}")
        
        return verification
    
    def load_history(self) -> Dict[str, Dict[str, Any]]:
        """Load job history from file.
        
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of job IDs to job results
        """
        try:
            if os.path.exists(self.results_file):
                print(f"Loading history from {self.results_file}...")
                if self.workflow.load_results(self.results_file):
                    history = self.workflow.get_job_history()
                    print(f"Loaded {len(history)} job records")
                    return history
                else:
                    print("Failed to load history")
                    return {}
            else:
                print("No history file found")
                return {}
        except Exception as e:
            print(f"Error loading history: {e}")
            return {}
    
    def show_history(self) -> None:
        """Display job history."""
        history = self.workflow.get_job_history()
        
        if not history:
            history = self.load_history()
        
        if history:
            print("\nJob History:")
            for job_id, result in history.items():
                mol_id = result.get("molecule_id", "unknown")
                binding = result.get("binding_affinity", "N/A")
                toxicity = result.get("toxicity_score", "N/A")
                
                verification = "✓" if "verification" in result and "tx_hash" in result["verification"] else "✗"
                timestamp = result.get("submission_time", "N/A")
                
                print(f"  {job_id[:8]}... | {mol_id} | BA: {binding} | Tox: {toxicity} | {verification} | {timestamp}")
        else:
            print("No job history available")


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Verifiable Drug Screening System with Purechain Blockchain")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Connect command
    connect_parser = subparsers.add_parser("connect", help="Connect to blockchain")
    
    # Screen command
    screen_parser = subparsers.add_parser("screen", help="Screen a single molecule")
    screen_parser.add_argument("molecule_id", help="Identifier for the molecule")
    screen_parser.add_argument("smiles", help="SMILES string representation of the molecule")
    screen_parser.add_argument("--target", "-t", dest="target_id", default="default",
                              help="Identifier for the target protein")
    
    # Batch screen command
    batch_parser = subparsers.add_parser("batch", help="Run batch screening from file")
    batch_parser.add_argument("input_file", help="JSON file with molecule data")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a screening result")
    verify_parser.add_argument("job_id", help="Job ID to verify")
    verify_parser.add_argument("tx_hash", help="Transaction hash from blockchain")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show job history")
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI application."""
    args = parse_arguments()
    cli = DrugScreeningCLI()
    
    if args.command == "connect":
        cli.connect()
    elif args.command == "screen":
        cli.screen(args.molecule_id, args.smiles, args.target_id)
    elif args.command == "batch":
        cli.batch_screen(args.input_file)
    elif args.command == "verify":
        cli.verify(args.job_id, args.tx_hash)
    elif args.command == "history":
        cli.show_history()
    else:
        # Default: show help
        print("Verifiable Drug Screening System")
        print("=================================")
        print("Commands:")
        print("  connect             Connect to blockchain")
        print("  screen <id> <smiles> Screen a single molecule")
        print("  batch <file>        Run batch screening from file")
        print("  verify <id> <hash>  Verify a screening result")
        print("  history             Show job history")
        print("\nUse -h with any command for more info")


if __name__ == "__main__":
    main()
