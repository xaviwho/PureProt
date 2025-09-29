#!/usr/bin/env python3
"""
Real Data Analysis - Based on Actual Screening Results
Analyzes the actual screening_results.csv with 100 molecules for paper enhancement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def analyze_real_screening_results():
    """Analyze the actual screening results from screening_results.csv"""
    print("=== Analyzing Real Screening Results ===")
    
    df = pd.read_csv('screening_results.csv')
    
    # Basic statistics
    total_molecules = len(df)
    lipinski_pass = df['passes_lipinski'].sum()
    lipinski_rate = (lipinski_pass / total_molecules) * 100
    
    pic50_stats = {
        'mean': df['predicted_pIC50'].mean(),
        'std': df['predicted_pIC50'].std(),
        'min': df['predicted_pIC50'].min(),
        'max': df['predicted_pIC50'].max(),
        'median': df['predicted_pIC50'].median()
    }
    
    # Molecular property statistics
    prop_stats = {
        'molecular_weight': {
            'mean': df['molecular_weight'].mean(),
            'std': df['molecular_weight'].std(),
            'range': (df['molecular_weight'].min(), df['molecular_weight'].max())
        },
        'logp': {
            'mean': df['logp'].mean(),
            'std': df['logp'].std(),
            'range': (df['logp'].min(), df['logp'].max())
        },
        'hbd': {
            'mean': df['h_bond_donors'].mean(),
            'std': df['h_bond_donors'].std(),
            'range': (df['h_bond_donors'].min(), df['h_bond_donors'].max())
        },
        'hba': {
            'mean': df['h_bond_acceptors'].mean(),
            'std': df['h_bond_acceptors'].std(),
            'range': (df['h_bond_acceptors'].min(), df['h_bond_acceptors'].max())
        }
    }
    
    print(f"Dataset: {total_molecules} molecules targeting {df['target_id'].iloc[0]}")
    print(f"Lipinski compliance: {lipinski_pass}/{total_molecules} ({lipinski_rate:.1f}%)")
    print(f"pIC50 statistics:")
    print(f"  Mean: {pic50_stats['mean']:.3f} ± {pic50_stats['std']:.3f}")
    print(f"  Range: {pic50_stats['min']:.2f} - {pic50_stats['max']:.2f}")
    print(f"  Median: {pic50_stats['median']:.3f}")
    
    return df, pic50_stats, prop_stats, lipinski_rate

def create_real_data_plots(df, pic50_stats, prop_stats):
    """Create plots based on actual screening data"""
    print("=== Creating Real Data Visualizations ===")
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Screening Results Analysis - Amyloid-β A4 Protein (100 Molecules)', 
                 fontsize=16, fontweight='bold')
    
    # 1. pIC50 distribution
    axes[0,0].hist(df['predicted_pIC50'], bins=15, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0,0].axvline(pic50_stats['mean'], color='red', linestyle='--', linewidth=2, 
                     label=f'Mean: {pic50_stats["mean"]:.2f}')
    axes[0,0].set_title('Predicted pIC50 Distribution', fontweight='bold')
    axes[0,0].set_xlabel('Predicted pIC50')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # 2. Lipinski compliance
    lipinski_counts = df['passes_lipinski'].value_counts()
    colors = ['lightcoral', 'lightgreen']
    axes[0,1].pie(lipinski_counts.values, labels=['Fails Lipinski', 'Passes Lipinski'], 
                 colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title('Lipinski Rule of Five Compliance', fontweight='bold')
    
    # 3. Molecular weight vs pIC50
    scatter = axes[0,2].scatter(df['molecular_weight'], df['predicted_pIC50'], 
                               c=df['passes_lipinski'], cmap='RdYlGn', alpha=0.7, s=60)
    axes[0,2].set_title('Molecular Weight vs pIC50', fontweight='bold')
    axes[0,2].set_xlabel('Molecular Weight (Da)')
    axes[0,2].set_ylabel('Predicted pIC50')
    axes[0,2].grid(alpha=0.3)
    
    # 4. LogP distribution
    axes[1,0].hist(df['logp'], bins=15, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    axes[1,0].axvline(prop_stats['logp']['mean'], color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {prop_stats["logp"]["mean"]:.2f}')
    axes[1,0].set_title('LogP Distribution', fontweight='bold')
    axes[1,0].set_xlabel('LogP')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 5. H-bond donors vs acceptors
    axes[1,1].scatter(df['h_bond_donors'], df['h_bond_acceptors'], 
                     c=df['predicted_pIC50'], cmap='viridis', alpha=0.7, s=60)
    axes[1,1].set_title('H-Bond Donors vs Acceptors', fontweight='bold')
    axes[1,1].set_xlabel('H-Bond Donors')
    axes[1,1].set_ylabel('H-Bond Acceptors')
    axes[1,1].grid(alpha=0.3)
    
    # 6. Lipinski violations distribution
    viol_counts = df['lipinski_violations'].value_counts().sort_index()
    axes[1,2].bar(viol_counts.index, viol_counts.values, alpha=0.7, color='orange', edgecolor='darkorange')
    axes[1,2].set_title('Lipinski Violations Distribution', fontweight='bold')
    axes[1,2].set_xlabel('Number of Violations')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_screening_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Real data analysis plot saved as 'real_screening_analysis.png'")
    
    return fig

def generate_paper_table():
    """Generate table for paper based on real data"""
    print("=== Generating Paper Table ===")
    
    df = pd.read_csv('screening_results.csv')
    
    # Get top 10 compounds by pIC50
    top_10 = df.nlargest(10, 'predicted_pIC50')[['molecule_id', 'predicted_pIC50', 'molecular_weight', 
                                                  'logp', 'h_bond_donors', 'h_bond_acceptors', 'passes_lipinski']]
    
    print("\nTABLE: Top 10 Screened Drug-Like Candidates")
    print("=" * 80)
    print(f"{'ChEMBL ID':<15} {'pIC50':<8} {'MW':<8} {'LogP':<8} {'HBD':<5} {'HBA':<5} {'Lipinski':<8}")
    print("-" * 80)
    
    for _, row in top_10.iterrows():
        lipinski_status = "Pass" if row['passes_lipinski'] else "Fail"
        print(f"{row['molecule_id']:<15} {row['predicted_pIC50']:<8.2f} {row['molecular_weight']:<8.1f} "
              f"{row['logp']:<8.2f} {row['h_bond_donors']:<5.0f} {row['h_bond_acceptors']:<5.0f} {lipinski_status:<8}")
    
    # Save to CSV
    top_10.to_csv('top_10_screened_compounds.csv', index=False)
    print(f"\n✅ Top 10 compounds saved to 'top_10_screened_compounds.csv'")
    
    return top_10

def calculate_screening_metrics():
    """Calculate key metrics for paper"""
    print("=== Calculating Screening Metrics ===")
    
    df = pd.read_csv('screening_results.csv')
    
    metrics = {
        'total_screened': len(df),
        'target_protein': df['target_id'].iloc[0],
        'lipinski_compliance_rate': (df['passes_lipinski'].sum() / len(df)) * 100,
        'high_affinity_compounds': len(df[df['predicted_pIC50'] > 6.0]),
        'drug_like_high_affinity': len(df[(df['predicted_pIC50'] > 6.0) & (df['passes_lipinski'] == True)]),
        'pic50_statistics': {
            'mean': df['predicted_pIC50'].mean(),
            'std': df['predicted_pIC50'].std(),
            'min': df['predicted_pIC50'].min(),
            'max': df['predicted_pIC50'].max()
        },
        'molecular_diversity': {
            'mw_range': (df['molecular_weight'].min(), df['molecular_weight'].max()),
            'logp_range': (df['logp'].min(), df['logp'].max()),
            'unique_scaffolds': len(df['molecule_id'].unique())
        }
    }
    
    print(f"Screening Summary:")
    print(f"  Total molecules screened: {metrics['total_screened']}")
    print(f"  Target: {metrics['target_protein']}")
    print(f"  Lipinski compliance: {metrics['lipinski_compliance_rate']:.1f}%")
    print(f"  High affinity (pIC50 > 6.0): {metrics['high_affinity_compounds']}")
    print(f"  Drug-like + high affinity: {metrics['drug_like_high_affinity']}")
    print(f"  pIC50 range: {metrics['pic50_statistics']['min']:.2f} - {metrics['pic50_statistics']['max']:.2f}")
    
    return metrics

def main():
    """Main analysis function"""
    print("🔬 Real Data Analysis - Binding Affinity Prediction Results")
    print("=" * 70)
    
    # Analyze real screening results
    df, pic50_stats, prop_stats, lipinski_rate = analyze_real_screening_results()
    
    # Create visualizations
    create_real_data_plots(df, pic50_stats, prop_stats)
    
    # Generate paper components
    top_10 = generate_paper_table()
    metrics = calculate_screening_metrics()
    
    print(f"\n🎯 PAPER COMPONENTS GENERATED:")
    print(f"✅ Real screening data analyzed: {len(df)} molecules")
    print(f"✅ Lipinski compliance: {lipinski_rate:.1f}%")
    print(f"✅ pIC50 range: {pic50_stats['min']:.2f} - {pic50_stats['max']:.2f}")
    print(f"✅ Top 10 compounds table created")
    print(f"✅ Comprehensive visualization generated")
    
    print(f"\n📊 Generated Files:")
    print(f"  - real_screening_analysis.png")
    print(f"  - top_10_screened_compounds.csv")

if __name__ == "__main__":
    main()
