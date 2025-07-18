# -*- coding: utf-8 -*-
"""Module for fetching and preparing bioactivity data from ChEMBL.

This module provides functions to download data for a given ChEMBL target ID,
process it by cleaning SMILES strings and calculating pIC50 values, and save
it to a CSV file for model training.
"""

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm


def get_bioactivities(target_id: str) -> pd.DataFrame:
    """Fetch bioactivity data for a specific ChEMBL target ID.

    Args:
        target_id: The ChEMBL ID of the target.

    Returns:
        A pandas DataFrame containing the bioactivity data.
    """
    print(f"Fetching bioactivities for target: {target_id}...")
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(res)
    print(f"Found {len(df)} bioactivity records.")
    return df


def process_bioactivities(df: pd.DataFrame) -> pd.DataFrame:
    """Process the raw bioactivity data.

    This function performs the following steps:
    1. Drops records with missing standard_value or canonical_smiles.
    2. Converts standard_value to numeric, coercing errors.
    3. Calculates pIC50 values.
    4. Keeps only essential columns: canonical_smiles and pIC50.

    Args:
        df: DataFrame with raw bioactivity data.

    Returns:
        A processed and cleaned DataFrame.
    """
    print("Processing bioactivity data...")
    df.dropna(subset=["standard_value", "canonical_smiles"], inplace=True)
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors='coerce')
    df.dropna(subset=["standard_value"], inplace=True)

    # Calculate pIC50
    df["pIC50"] = -np.log10(df["standard_value"] * 1e-9)
    df.rename(columns={'canonical_smiles': 'smiles'}, inplace=True)

    processed_df = df[["smiles", "pIC50"]]
    print(f"Processed data contains {len(processed_df)} records.")
    return processed_df


def fetch_and_prepare_data(target_id: str, output_path: str):
    """Main function to fetch, process, and save ChEMBL data."""
    raw_data = get_bioactivities(target_id)
    if raw_data.empty:
        print("No data found for the specified target.")
        return

    processed_data = process_bioactivities(raw_data)
    
    print(f"Saving prepared data to: {output_path}")
    processed_data.to_csv(output_path, index=False)
    print("Data preparation complete.")
