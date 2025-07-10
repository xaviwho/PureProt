import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import joblib
import pathlib

# Define paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DATA_FILE = BASE_DIR / "data" / "training_data.csv"
MODEL_FILE = BASE_DIR / "binding_affinity_model.joblib"

def train_and_save_model():
    """
    Loads the bioactivity data, trains a Random Forest model to predict pIC50,
    evaluates its performance, and saves the trained model to a file.
    """
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_FILE}")
        return

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    # Generate molecular fingerprints from SMILES strings
    print("Generating Morgan fingerprints from SMILES...")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def smiles_to_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Using a radius of 2 and 2048 bits is a common standard
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    df['fingerprint'] = df['canonical_smiles'].apply(smiles_to_fp)
    
    # Remove molecules that could not be processed
    df = df.dropna(subset=['fingerprint'])

    X = list(df['fingerprint'])
    target = 'pIC50'
    y = df[target]

    # Split data into training and testing sets (80% train, 20% test)
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Support Vector Regressor (SVR) model
    print("Training the SVR model...")
    # SVR with a radial basis function (RBF) kernel is a good default
    model = SVR()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model on the testing set
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Model Performance on Test Set:")
    print(f"R-squared (R²): {r2:.4f}")
    print("-" * 30)
    print("\nAn R-squared value close to 1.0 indicates that the model predicts the data well.")


    # Save the trained model to a file
    print(f"Saving the trained model to {MODEL_FILE}...")
    joblib.dump(model, MODEL_FILE)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
