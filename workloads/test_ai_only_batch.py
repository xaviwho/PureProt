#!/usr/bin/env python3
"""Test script for AI-only batch screening functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PureProt import PureProtCLI
import pandas as pd

def test_ai_only_batch():
    """Test the AI-only batch screening with correct model path"""
    print("Testing AI-only batch screening...")
    
    # Initialize CLI
    cli = PureProtCLI()
    
    # Test with top 10 compounds and correct model path
    csv_path = "top_10_compounds.csv"
    model_path = "modeling/binding_affinity_model.joblib"
    output_path = "test_ai_only_results.csv"
    
    print(f"Input CSV: {csv_path}")
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV {csv_path} not found")
        return False
        
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return False
    
    try:
        # Run the batch screening
        cli.run_batch(csv_path, model_path, output_path)
        
        # Check if output was created
        if os.path.exists(output_path):
            print(f"Success! Output file created: {output_path}")
            
            # Read and display first few rows
            df = pd.read_csv(output_path)
            print(f"Output contains {len(df)} rows")
            print("First few rows:")
            print(df.head())
            return True
        else:
            print("Error: Output file was not created")
            return False
            
    except Exception as e:
        print(f"Error during batch screening: {e}")
        return False

if __name__ == "__main__":
    success = test_ai_only_batch()
    if success:
        print("\n✅ AI-only batch screening test PASSED")
    else:
        print("\n❌ AI-only batch screening test FAILED")
