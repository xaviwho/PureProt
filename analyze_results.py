#!/usr/bin/env python3
"""
Analyze Screening Results

This script reads the 'screening_results.csv' file, identifies the top 10
drug candidates based on Lipinski's Rule of Five and predicted pIC50,
and prints a summary table in Markdown format suitable for reports.
"""

import pandas as pd
import argparse
from pathlib import Path

def analyze_results(file_path: Path, top_n: int = 10):
    """
    Reads screening results, filters for drug-like molecules, ranks them,
    and prints a summary table of the top candidates.

    Args:
        file_path (Path): The path to the screening_results.csv file.
        top_n (int): The number of top candidates to report.
    """
    if not file_path.exists():
        print(f"Error: Results file not found at {file_path}")
        return

    print(f"Loading screening results from {file_path}...")
    df = pd.read_csv(file_path)

    # --- Step 1: Filter for potential drug candidates ---
    # We only want molecules that pass Lipinski's Rule of Five.
    lipinski_pass_df = df[df['passes_lipinski'] == True].copy()
    
    if lipinski_pass_df.empty:
        print("No molecules passed Lipinski's Rule of Five. No top candidates to report.")
        return

    print(f"Found {len(lipinski_pass_df)} molecules that passed Lipinski's Rule of Five.")

    # --- Step 2: Rank candidates by binding affinity ---
    # Sort by 'predicted_pIC50' in descending order (higher is better).
    top_candidates = lipinski_pass_df.sort_values(by='predicted_pIC50', ascending=False)

    # --- Step 3: Select the top N candidates for the report ---
    report_df = top_candidates.head(top_n)

    # --- Step 4: Format and print the report table ---
    print(f"\n--- Top {len(report_df)} Drug Candidates ---\
")
    
    # Select and reorder columns for the final report
    report_columns = [
        'molecule_id', 
        'predicted_pIC50', 
        'molecular_weight', 
        'logp', 
        'h_bond_donors', 
        'h_bond_acceptors'
    ]
    
    # Ensure all columns exist before trying to select them
    final_df = report_df[[col for col in report_columns if col in report_df.columns]].copy()

    # Round numeric values for cleaner presentation
    final_df['predicted_pIC50'] = final_df['predicted_pIC50'].round(2)
    final_df['molecular_weight'] = final_df['molecular_weight'].round(2)
    final_df['logp'] = final_df['logp'].round(2)

    # Rename columns for the markdown table
    final_df.rename(columns={
        'molecule_id': 'ChEMBL ID',
        'predicted_pIC50': 'Predicted pIC50',
        'molecular_weight': 'Mol. Weight',
        'logp': 'LogP',
        'h_bond_donors': 'H-Bond Donors',
        'h_bond_acceptors': 'H-Bond Acceptors'
    }, inplace=True)


    print("The following table can be copied directly into your publication or report:")
    print(final_df.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze drug screening results.")
    parser.add_argument(
        "--file",
        type=str,
        default="screening_results.csv",
        help="Path to the screening results CSV file."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to display in the report."
    )
    args = parser.parse_args()
    
    analyze_results(Path(args.file), args.top)
