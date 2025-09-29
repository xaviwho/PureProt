#!/usr/bin/env python3
"""
Extract AI-only predictions from hybrid results for comparative analysis
"""

import pandas as pd
import sys

def extract_ai_only_results():
    """Extract AI predictions from hybrid screening results"""
    
    # Load hybrid results
    hybrid_file = "natural_products_for_screening_hybrid_results.csv"
    try:
        df = pd.read_csv(hybrid_file)
        print(f"Loaded {len(df)} compounds from hybrid results")
    except FileNotFoundError:
        print(f"Error: {hybrid_file} not found")
        return
    
    # Extract AI-only columns
    ai_columns = [
        'molecule_id', 'smiles', 'target_id', 'predicted_pIC50', 'ai_status',
        'molecular_weight', 'logp', 'hbd', 'hba', 'psa', 'rotatable_bonds',
        'lipinski_violations', 'drug_likeness_score', 'drug_likeness_status'
    ]
    
    # Create AI-only dataframe
    ai_only_df = df[ai_columns].copy()
    
    # Sort by AI prediction (highest first)
    ai_only_df = ai_only_df.sort_values('predicted_pIC50', ascending=False)
    
    # Save AI-only results
    output_file = "natural_products_ai_only_results.csv"
    ai_only_df.to_csv(output_file, index=False)
    
    print(f"AI-only results saved to: {output_file}")
    
    # Display top 10 predictions
    print("\nTop 10 AI Predictions:")
    print("=" * 60)
    top_10 = ai_only_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"{row['molecule_id']:<40} pIC50: {row['predicted_pIC50']:.3f}")
    
    # Statistics
    print(f"\nAI Prediction Statistics:")
    print(f"Total compounds: {len(ai_only_df)}")
    print(f"Prediction range: {ai_only_df['predicted_pIC50'].min():.3f} - {ai_only_df['predicted_pIC50'].max():.3f}")
    print(f"Mean prediction: {ai_only_df['predicted_pIC50'].mean():.3f}")
    print(f"Compounds with pIC50 > 6.0: {len(ai_only_df[ai_only_df['predicted_pIC50'] > 6.0])}")

if __name__ == "__main__":
    extract_ai_only_results()
