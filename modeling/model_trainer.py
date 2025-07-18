# -*- coding: utf-8 -*-
"""Module for training and saving the machine learning model.

This module handles the loading of prepared data, generation of molecular
fingerprints, training of a Support Vector Regression (SVR) model, and
saving the trained model to a file.
"""

import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

def generate_fingerprints(smiles_series: pd.Series) -> np.ndarray:
    """Generate Morgan fingerprints from a series of SMILES strings."""
    fingerprints = []
    print("Generating Morgan fingerprints...")
    for smiles in tqdm(smiles_series):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints.append(np.array(fp))
    return np.array(fingerprints)

def train_model(X: np.ndarray, y: pd.Series) -> SVR:
    """Train an SVR model and print performance metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training SVR model on {len(X_train)} samples...")
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Training Complete ---")
    print("Model Performance (on test set):")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  R-squared (R2) Score: {r2:.4f}")
    
    return model

def train_and_save_model(dataset_path: str, model_output_path: str):
    """Load data, train a model, and save it to a file."""
    print(f"Loading dataset from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        df.dropna(subset=['smiles', 'pIC50'], inplace=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    X = generate_fingerprints(df['smiles'])
    y = df['pIC50']

    if len(X) == 0:
        print("No valid molecules found to train the model.")
        return

    model = train_model(X, y)
    
    print(f"\nSaving trained model to: {model_output_path}")
    joblib.dump(model, model_output_path)
    print("Model saved successfully.")
