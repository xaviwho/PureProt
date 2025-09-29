"""
Consensus AI Model Module for PureProtX

This module implements a true Consensus AI system using ensemble methods
with SVR, Random Forest, and Gradient Boosting models.
"""

import os
import joblib
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


class ConsensusAIModel:
    """
    Consensus AI Model implementing ensemble of SVR, Random Forest, and Gradient Boosting
    for robust molecular property prediction.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Consensus AI Model.
        
        Args:
            model_path: Path to saved ensemble model file
        """
        self.models = {
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_hash = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def calculate_molecular_features(self, smiles: str) -> np.ndarray:
        """
        Calculate molecular features from SMILES string.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            Feature vector as numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Calculate molecular descriptors
        features = []
        
        # Basic descriptors
        features.extend([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.FractionCsp3(mol),
            Descriptors.BalabanJ(mol)
        ])
        
        # Morgan fingerprint (2048 bits)
        fingerprint = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features.extend(list(fingerprint))
        
        return np.array(features)
    
    def prepare_dataset(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset for training from CSV file.
        
        Args:
            data_path: Path to CSV file with SMILES and pIC50 columns
            
        Returns:
            Tuple of (features, targets)
        """
        df = pd.read_csv(data_path)
        
        if 'smiles' not in df.columns or 'pic50' not in df.columns:
            raise ValueError("Dataset must contain 'smiles' and 'pic50' columns")
        
        features = []
        targets = []
        
        for _, row in df.iterrows():
            try:
                feat = self.calculate_molecular_features(row['smiles'])
                features.append(feat)
                targets.append(row['pic50'])
            except Exception as e:
                print(f"Skipping molecule {row.get('molecule_id', 'unknown')}: {e}")
                continue
        
        return np.array(features), np.array(targets)
    
    def train(self, data_path: str, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the consensus AI model ensemble.
        
        Args:
            data_path: Path to training dataset
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary of performance metrics
        """
        print("Preparing dataset...")
        X, y = self.prepare_dataset(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train each model in the ensemble
        print("Training ensemble models...")
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate individual model
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            model_scores[name] = {'r2': r2, 'rmse': rmse}
            print(f"    {name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        
        # Evaluate consensus model
        consensus_pred = self.predict_consensus(X_test_scaled)
        consensus_r2 = r2_score(y_test, consensus_pred)
        consensus_rmse = np.sqrt(mean_squared_error(y_test, consensus_pred))
        
        model_scores['consensus'] = {'r2': consensus_r2, 'rmse': consensus_rmse}
        print(f"  Consensus AI: R² = {consensus_r2:.4f}, RMSE = {consensus_rmse:.4f}")
        
        self.is_trained = True
        return model_scores
    
    def predict_consensus(self, X: np.ndarray) -> np.ndarray:
        """
        Make consensus predictions using all models in ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Consensus predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Simple average consensus
        consensus = np.mean(predictions, axis=0)
        return consensus
    
    def predict_single(self, smiles: str) -> Dict[str, float]:
        """
        Predict pIC50 for a single molecule.
        
        Args:
            smiles: SMILES string of molecule
            
        Returns:
            Dictionary with individual and consensus predictions
        """
        features = self.calculate_molecular_features(smiles)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        results = {}
        for name, model in self.models.items():
            results[name] = float(model.predict(features_scaled)[0])
        
        # Calculate consensus
        results['consensus'] = np.mean(list(results.values()))
        
        return results
    
    def save_model(self, output_path: str) -> str:
        """
        Save the trained ensemble model.
        
        Args:
            output_path: Path to save model file
            
        Returns:
            SHA-256 hash of saved model file
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, output_path)
        
        # Calculate and store model hash
        with open(output_path, 'rb') as f:
            model_bytes = f.read()
            self.model_hash = hashlib.sha256(model_bytes).hexdigest()
        
        print(f"Consensus AI model saved to: {output_path}")
        print(f"Model hash: {self.model_hash}")
        
        return self.model_hash
    
    def load_model(self, model_path: str) -> str:
        """
        Load a trained ensemble model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            SHA-256 hash of loaded model file
        """
        model_data = joblib.load(model_path)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
            self.model_hash = hashlib.sha256(model_bytes).hexdigest()
        
        print(f"Consensus AI model loaded from: {model_path}")
        print(f"Model hash: {self.model_hash}")
        
        return self.model_hash
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'is_trained': self.is_trained,
            'model_hash': self.model_hash,
            'models': list(self.models.keys()),
            'consensus_method': 'simple_average'
        }
