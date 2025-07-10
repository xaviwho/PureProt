import pandas as pd
import requests
import time
import joblib
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator, rdMolDescriptors

# --- Configuration ---
BASE_DIR = Path(__file__).parent.resolve()
DATA_FILE = BASE_DIR / "data" / "training_data.csv"
TEMP_MODEL_DIR = BASE_DIR / "temp_models"

# --- Model Definitions ---
models_to_evaluate = {
    "RF": RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True),
    "GB": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "MLP": MLPRegressor(random_state=42, max_iter=500),
    "Ridge": Ridge(random_state=42)
}

def get_model_size(model_path):
    """Calculate the size of a file in kilobytes."""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / 1024

# Create the generator once and reuse it for efficiency
morgan_fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smiles):
    """Converts a SMILES string to a Morgan fingerprint using the modern RDKit API."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return morgan_fp_generator.GetFingerprint(mol)
    except:
        return None

def prepare_training_data():
    """Download and prepare training data from ChEMBL if not present."""
    if DATA_FILE.exists():
        print(f"Found existing training data at {DATA_FILE}")
        return

    print(f"Training data not found. Downloading from ChEMBL...")
    # Use the new API endpoint format for ChEMBL
    # Use the JSON endpoint for reliability
    url = 'https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL2487&pchembl_value__isnull=false&limit=1000'
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()

        # Extract the 'activities' list from the JSON response
        activities = data.get('activities', [])
        if not activities:
            print("Error: No activities found in the API response.")
            return

        df = pd.DataFrame(activities)
        
        # Data cleaning and preparation
        # Ensure the required columns are present
        if 'canonical_smiles' not in df.columns or 'pchembl_value' not in df.columns:
            print(f"Error: Required columns not in response. Available columns: {df.columns.tolist()}")
            return
            
        df = df[['canonical_smiles', 'pchembl_value']]
        df.rename(columns={'pchembl_value': 'pIC50'}, inplace=True)
        df['pIC50'] = pd.to_numeric(df['pIC50'], errors='coerce')
        df = df.dropna()
        df.to_csv(DATA_FILE, index=False)
        print(f"Successfully downloaded and saved training data to {DATA_FILE}")
    except Exception as e:
        print(f"Failed to download or process training data: {e}")
        raise

def evaluate_models():
    """Train, evaluate, and compare multiple regression models."""
    # --- 1. Load and Prepare Data ---
    try:
        prepare_training_data()
        print("Loading and preparing dataset...")
        df = pd.read_csv(DATA_FILE)
    except:
        return None

    df['fingerprint'] = df['canonical_smiles'].apply(smiles_to_fp)
    df = df.dropna(subset=['fingerprint', 'pIC50'])
    df = df.drop_duplicates(subset=['canonical_smiles'])
    X = list(df['fingerprint'])
    y = df['pIC50']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data preparation complete.")

    # --- 2. Train and Evaluate Models ---
    results = []
    TEMP_MODEL_DIR.mkdir(exist_ok=True)

    for name, model in models_to_evaluate.items():
        print(f"\n--- Evaluating {name} ---")

        # Train model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Save model and get size
        model_path = TEMP_MODEL_DIR / f"{name}_model.joblib"
        joblib.dump(model, model_path)
        model_size = get_model_size(model_path)
        os.remove(model_path) # Clean up temporary model file

        results.append({
            "Model": name,
            "R-squared": r2,
            "MSE": mse,
            "MAE": mae,
            "Training Time (s)": training_time,
            "Model Size (KB)": model_size
        })
        print(f"{name} evaluation complete.")

    # Clean up temp directory
    os.rmdir(TEMP_MODEL_DIR)

    # --- 3. Display Results ---
    results_df = pd.DataFrame(results)
    print("\n--- Comparative Model Performance ---")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

if __name__ == "__main__":
    evaluate_models()
