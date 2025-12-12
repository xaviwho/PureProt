import csv
import os

def convert_smi_to_csv(smi_path, csv_path):
    """Converts a .smi file to a .csv file compatible with PureProt.

    Args:
        smi_path (str): The full path to the input .smi file.
        csv_path (str): The full path for the output .csv file.
    """
    print(f"Reading from: {smi_path}")
    
    try:
        with open(smi_path, 'r') as f_in, open(csv_path, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            # Write the header row required by PureProt
            writer.writerow(['molecule_id', 'smiles'])
            
            count = 0
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                # The .smi file is tab-separated with SMILES first, then ID
                parts = line.split('\t')
                if len(parts) == 2:
                    smiles, molecule_id = parts[0], parts[1]
                    writer.writerow([molecule_id.strip(), smiles.strip()])
                    count += 1
                else:
                    print(f"Warning: Skipping malformed line: {line}")
            
            print(f"Successfully converted {count} molecules.")
            print(f"Output saved to: {csv_path}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {smi_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Update these paths to match your files.
    
    # Path to the .smi file you downloaded
    INPUT_SMI_FILE = r"C:\Users\Xavie\Downloads\SMILES\SMILES\SMILES__East_Africa\smiles_unique_EANPDB.smi"
    
    # Desired path for the output CSV file (will be created in the current directory)
    OUTPUT_CSV_FILE = "natural_products_for_screening.csv"
    
    # --- Run Conversion ---
    if os.path.exists(INPUT_SMI_FILE):
        convert_smi_to_csv(INPUT_SMI_FILE, OUTPUT_CSV_FILE)
    else:
        print(f"Error: Input file not found at the specified path.")
        print("Please update the 'INPUT_SMI_FILE' variable in this script.")
