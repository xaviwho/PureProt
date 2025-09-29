#!/usr/bin/env python3
"""Test script for comparison analysis functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PureProt import PureProtCLI
import pandas as pd

def test_comparison_analysis():
    """Test the comparison analysis functionality"""
    print("Testing comparison analysis...")
    
    # Initialize CLI
    cli = PureProtCLI()
    
    # Test files
    ai_results = "top_10_ai_only_corrected.csv"
    hybrid_results = "top_10_compounds_hybrid_results.csv"
    output_path = "test_comparison_results.csv"
    
    print(f"AI results: {ai_results}")
    print(f"Hybrid results: {hybrid_results}")
    print(f"Output path: {output_path}")
    
    # Check if input files exist
    if not os.path.exists(ai_results):
        print(f"Error: AI results file {ai_results} not found")
        return False
        
    if not os.path.exists(hybrid_results):
        print(f"Error: Hybrid results file {hybrid_results} not found")
        return False
    
    try:
        # Run the comparison analysis
        cli.run_compare_results(ai_results, None, hybrid_results, output_path)
        
        # Check if output was created
        if os.path.exists(output_path):
            print(f"Success! Comparison output created: {output_path}")
            
            # Read and display results
            df = pd.read_csv(output_path)
            print(f"Comparison contains {len(df)} molecules")
            print("Comparison results:")
            print(df.head())
            return True
        else:
            print("Error: Comparison output file was not created")
            return False
            
    except Exception as e:
        print(f"Error during comparison analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comparison_analysis()
    if success:
        print("\n✅ Comparison analysis test PASSED")
    else:
        print("\n❌ Comparison analysis test FAILED")
