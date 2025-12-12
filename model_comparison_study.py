#!/usr/bin/env python3
"""
Model Comparison Study for PureProtX Consensus AI Selection
Evaluates 6+ machine learning models to justify the selection of 
SVR, Random Forest, and Gradient Boosting for the consensus ensemble.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# ML models to compare
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pureprot.ai_model import ConsensusAIModel


class ModelComparisonStudy:
    """
    Comprehensive comparison of ML models for molecular property prediction.
    
    FAIRNESS MEASURES:
    - All models use sklearn default parameters or commonly accepted values
    - Same n_estimators (100) for all ensemble methods
    - Same random_state for reproducibility
    - Same train/test split for all models
    - Same feature scaling applied to all models
    - Same evaluation metrics for all models
    - No hyperparameter tuning (to avoid bias toward any specific model)
    """
    
    def __init__(self, random_state=42):
        """Initialize the comparison study."""
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Define all models to compare with UNBIASED default parameters
        # All models use sklearn defaults or commonly accepted values
        self.models = {
            # Proposed Consensus Models
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1),
            'Random_Forest': RandomForestRegressor(n_estimators=100, max_depth=None, 
                                                   min_samples_split=2, random_state=random_state),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                           max_depth=3, random_state=random_state),
            
            # Alternative Ensemble Methods (same n_estimators for fair comparison)
            'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=1.0, 
                                         random_state=random_state),
            'Extra_Trees': ExtraTreesRegressor(n_estimators=100, max_depth=None,
                                              min_samples_split=2, random_state=random_state),
            
            # Linear Models (default regularization)
            'Ridge': Ridge(alpha=1.0, random_state=random_state),
            'Lasso': Lasso(alpha=1.0, max_iter=1000, random_state=random_state),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, 
                                    random_state=random_state),
            
            # Instance-Based Learning
            'KNN': KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto'),
            
            # Neural Network (increased max_iter for convergence)
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu',
                               solver='adam', max_iter=1000, early_stopping=True,
                               random_state=random_state),
            
            # Simple Baseline
            'Decision_Tree': DecisionTreeRegressor(max_depth=None, min_samples_split=2,
                                                   random_state=random_state)
        }
        
        self.results = {}
        
    def load_and_prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare dataset using ConsensusAIModel feature extraction.
        
        Args:
            data_path: Path to CSV file with smiles and pIC50 columns
            
        Returns:
            Tuple of (features, targets)
        """
        print(f"Loading data from: {data_path}")
        
        # Use ConsensusAIModel for consistent feature extraction
        consensus_model = ConsensusAIModel()
        X, y = consensus_model.prepare_dataset(data_path)
        
        print(f"Dataset loaded: {len(X)} molecules, {X.shape[1]} features")
        return X, y
    
    def evaluate_single_model(self, name: str, model, X_train: np.ndarray, X_test: np.ndarray,
                             y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            name: Model name
            model: Model instance
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            
        Returns:
            Dictionary of performance metrics
        """
        print(f"  Evaluating {name}...")
        
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'model_name': name,
            
            # Test set metrics (primary)
            'test_r2': r2_score(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            
            # Training set metrics (check overfitting)
            'train_r2': r2_score(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            
            # Overfitting indicator
            'r2_gap': r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred),
            
            # Computational efficiency
            'train_time_sec': train_time,
            'predict_time_sec': predict_time,
            'predict_time_per_sample_ms': (predict_time / len(y_test)) * 1000
        }
        
        # Cross-validation (5-fold)
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='r2', n_jobs=-1)
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
        except Exception as e:
            print(f"    Warning: CV failed for {name}: {e}")
            metrics['cv_r2_mean'] = None
            metrics['cv_r2_std'] = None
        
        print(f"    Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}")
        
        return metrics
    
    def run_comparison(self, data_path: str, test_size: float = 0.2) -> pd.DataFrame:
        """
        Run comprehensive model comparison.
        
        Args:
            data_path: Path to dataset
            test_size: Fraction for test set
            
        Returns:
            DataFrame with comparison results
        """
        print("="*70)
        print("PureProtX Model Comparison Study")
        print("="*70)
        
        # Load data
        X, y = self.load_and_prepare_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"\nEvaluating {len(self.models)} models...\n")
        
        # Evaluate each model
        results_list = []
        for name, model in self.models.items():
            try:
                metrics = self.evaluate_single_model(
                    name, model, X_train_scaled, X_test_scaled, y_train, y_test
                )
                results_list.append(metrics)
            except Exception as e:
                print(f"  Error evaluating {name}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Sort by test R²
        results_df = results_df.sort_values('test_r2', ascending=False)
        
        self.results = results_df
        return results_df
    
    def evaluate_consensus_ensembles(self, data_path: str, test_size: float = 0.2):
        """
        Evaluate different consensus ensemble combinations.
        
        Args:
            data_path: Path to dataset
            test_size: Fraction for test set
        """
        print("\n" + "="*70)
        print("Consensus Ensemble Evaluation")
        print("="*70)
        
        # Load data
        X, y = self.load_and_prepare_data(data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define ensemble combinations to test
        ensembles = {
            'Proposed_Consensus': ['SVR', 'Random_Forest', 'Gradient_Boosting'],
            'All_Ensemble_Methods': ['Random_Forest', 'Gradient_Boosting', 'AdaBoost', 'Extra_Trees'],
            'Top_3_Individual': None,  # Will be determined from results
            'Linear_Ensemble': ['Ridge', 'Lasso', 'ElasticNet'],
            'Tree_Ensemble': ['Random_Forest', 'Extra_Trees', 'Decision_Tree']
        }
        
        # Get top 3 individual models
        if hasattr(self, 'results') and not self.results.empty:
            top_3_names = self.results.nlargest(3, 'test_r2')['model_name'].tolist()
            ensembles['Top_3_Individual'] = top_3_names
        
        ensemble_results = []
        
        for ensemble_name, model_names in ensembles.items():
            if model_names is None:
                continue
                
            print(f"\nEvaluating {ensemble_name}: {model_names}")
            
            # Train models and get predictions
            predictions = []
            for model_name in model_names:
                if model_name not in self.models:
                    print(f"  Warning: {model_name} not found, skipping")
                    continue
                    
                model = self.models[model_name]
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                predictions.append(pred)
            
            if len(predictions) == 0:
                continue
            
            # Calculate consensus (simple average)
            consensus_pred = np.mean(predictions, axis=0)
            
            # Calculate metrics
            metrics = {
                'ensemble_name': ensemble_name,
                'models': ', '.join(model_names),
                'n_models': len(predictions),
                'test_r2': r2_score(y_test, consensus_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, consensus_pred)),
                'test_mae': mean_absolute_error(y_test, consensus_pred)
            }
            
            ensemble_results.append(metrics)
            print(f"  R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}")
        
        self.ensemble_results = pd.DataFrame(ensemble_results)
        return self.ensemble_results
    
    def generate_report(self, output_dir: str = "."):
        """
        Generate comprehensive comparison report.
        
        Args:
            output_dir: Directory to save report files
        """
        print("\n" + "="*70)
        print("Generating Comparison Report")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to CSV
        results_file = os.path.join(output_dir, "model_comparison_results.csv")
        self.results.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        if hasattr(self, 'ensemble_results'):
            ensemble_file = os.path.join(output_dir, "ensemble_comparison_results.csv")
            self.ensemble_results.to_csv(ensemble_file, index=False)
            print(f"Ensemble results saved to: {ensemble_file}")
        
        # Generate summary statistics
        summary = {
            'best_model': self.results.iloc[0]['model_name'],
            'best_r2': float(self.results.iloc[0]['test_r2']),
            'best_rmse': float(self.results.iloc[0]['test_rmse']),
            'proposed_consensus_models': ['SVR', 'Random_Forest', 'Gradient_Boosting'],
            'proposed_consensus_ranks': [],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'fairness_measures': {
                'same_train_test_split': True,
                'same_feature_scaling': True,
                'same_random_state': self.random_state,
                'no_hyperparameter_tuning': True,
                'same_n_estimators_for_ensembles': 100,
                'sklearn_default_parameters': True
            }
        }
        
        # Get ranks of proposed models
        for model in summary['proposed_consensus_models']:
            rank = self.results[self.results['model_name'] == model].index[0] + 1
            summary['proposed_consensus_ranks'].append(int(rank))
        
        summary_file = os.path.join(output_dir, "comparison_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")
        
        # Print summary table
        print("\n" + "="*70)
        print("Model Comparison Summary (Top 10)")
        print("="*70)
        print(self.results[['model_name', 'test_r2', 'test_rmse', 'test_mae', 
                           'r2_gap', 'train_time_sec']].head(10).to_string(index=False))
        
        if hasattr(self, 'ensemble_results'):
            print("\n" + "="*70)
            print("Consensus Ensemble Comparison")
            print("="*70)
            print(self.ensemble_results[['ensemble_name', 'n_models', 'test_r2', 
                                        'test_rmse', 'test_mae']].to_string(index=False))
        
        print("\n" + "="*70)
        print("Justification for Proposed Consensus (SVR + RF + GB):")
        print("="*70)
        for i, model in enumerate(summary['proposed_consensus_models']):
            rank = summary['proposed_consensus_ranks'][i]
            row = self.results[self.results['model_name'] == model].iloc[0]
            print(f"{model}:")
            print(f"  Rank: #{rank}")
            print(f"  Test R²: {row['test_r2']:.4f}")
            print(f"  Test RMSE: {row['test_rmse']:.4f}")
            print(f"  Overfitting (R² gap): {row['r2_gap']:.4f}")
            print(f"  Training time: {row['train_time_sec']:.2f}s")
            print()


def main():
    """Main function to run model comparison study."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PureProtX Model Comparison Study")
    parser.add_argument("data_path", type=str, help="Path to dataset CSV file")
    parser.add_argument("--output", type=str, default=".", help="Output directory for results")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    
    args = parser.parse_args()
    
    # Run comparison study
    study = ModelComparisonStudy()
    
    # Compare individual models
    results = study.run_comparison(args.data_path, test_size=args.test_size)
    
    # Compare consensus ensembles
    ensemble_results = study.evaluate_consensus_ensembles(args.data_path, test_size=args.test_size)
    
    # Generate report
    study.generate_report(output_dir=args.output)
    
    print("\n✅ Model comparison study complete!")


if __name__ == "__main__":
    main()
