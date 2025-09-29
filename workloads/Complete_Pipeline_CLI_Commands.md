# Complete PureProt Pipeline CLI Commands

This guide provides all CLI commands to run the complete hybrid AI+docking+blockchain screening pipeline from start to finish.

## 🚀 **Quick Start: Complete Pipeline**

### **Option 1: Full Hybrid Screening (Recommended)**
```bash
# Complete hybrid AI+docking screening with blockchain verification
python PureProt.py hybrid-screen natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --protein 1hpv.pdb \
    --center "0.0,0.0,0.0" \
    --size "20,20,20"
```

### **Option 2: Docking-Only Screening**
```bash
# Molecular docking screening only
python PureProt.py dock natural_products_for_screening.csv \
    --protein 1hpv.pdb \
    --center "0.0,0.0,0.0" \
    --size "20,20,20"
```

### **Option 3: AI-Only Screening**
```bash
# AI-based screening only
python PureProt.py screen natural_products_for_screening.csv \
    --model egfr_model.joblib
```

---

## 📋 **Step-by-Step Complete Workflow**

### **Step 1: Environment Setup**
```bash
# Verify Python environment and dependencies
python --version
pip install -r requirements.txt

# Check PureProt CLI availability
python PureProt.py --help
```

### **Step 2: Data Preparation**
```bash
# Convert SMILES files to CSV format (if needed)
python PureProt.py convert input_molecules.smi output_molecules.csv

# Verify data format
head -5 natural_products_for_screening.csv
```

### **Step 3: Protein Preparation**
```bash
# Prepare protein structure for docking
python PureProt.py prep-protein 1hpv.pdb --output 1hpv_prepared.pdb

# Verify protein file exists
ls -la 1hpv*.pdb
```

### **Step 4: AI Model Training (Optional)**
```bash
# Fetch training data
python PureProt.py fetch-data --target EGFR --output egfr_data.csv

# Train AI model
python PureProt.py train-model egfr_data.csv --output egfr_model.joblib

# Verify model creation
ls -la *.joblib
```

### **Step 5: Molecular Docking Screening**
```bash
# Run molecular docking on natural products
python PureProt.py dock natural_products_for_screening.csv \
    --protein 1hpv.pdb \
    --center "0.0,0.0,0.0" \
    --size "20,20,20"
```

### **Step 6: Hybrid AI+Docking Screening**
```bash
# Run complete hybrid screening
python PureProt.py hybrid-screen natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --protein 1hpv.pdb \
    --center "0.0,0.0,0.0" \
    --size "20,20,20"
```

### **Step 7: Blockchain Verification**
```bash
# Verify results on blockchain
python PureProt.py verify --transaction-hash <hash_from_screening>

# View blockchain history
python PureProt.py history --limit 10
```

### **Step 8: Batch Processing**
```bash
# Process large datasets in batches
python PureProt.py batch natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --batch-size 100
```

### **Step 9: Performance Benchmarking**
```bash
# Generate performance benchmarks
python PureProt.py benchmark natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --iterations 5
```

---

## 🎯 **Specific Use Cases**

### **HIV-1 Protease (1HPV) Natural Products Screening**
```bash
# Complete pipeline for HIV protease inhibitor discovery
python PureProt.py hybrid-screen natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --protein 1hpv.pdb \
    --center "0.0,0.0,0.0" \
    --size "20,20,20"
```

### **EGFR Kinase Screening**
```bash
# EGFR-specific screening with custom binding site
python PureProt.py hybrid-screen natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --protein egfr_kinase.pdb \
    --center "25.0,30.0,15.0" \
    --size "20,20,20"
```

### **Custom Molecule Set**
```bash
# Screen custom molecules from SMILES file
python PureProt.py convert custom_molecules.smi custom_molecules.csv
python PureProt.py hybrid-screen custom_molecules.csv \
    --model braf_model.joblib \
    --protein target_protein.pdb \
    --center "10.0,15.0,20.0"
```

---

## ⚡ **Performance Optimization Commands**

### **Fast Screening (AI-Only)**
```bash
# Quick AI-based screening for large datasets
python PureProt.py screen natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --fast-mode
```

### **Parallel Processing**
```bash
# Use multiple cores for batch processing
python PureProt.py batch natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --workers 4 \
    --batch-size 50
```

### **Memory-Efficient Processing**
```bash
# Process large datasets with memory optimization
python PureProt.py batch natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --batch-size 25 \
    --memory-efficient
```

---

## 🔍 **Testing and Validation Commands**

### **System Status Check**
```bash
# Check docking engine status
python -c "
from modeling.advanced_docking_engine import create_docking_engine
engine = create_docking_engine()
print('Engine status:', engine.get_engine_status())
"
```

### **Single Molecule Test**
```bash
# Test with single molecule (ibuprofen)
python -c "
from modeling.advanced_docking_engine import create_docking_engine
engine = create_docking_engine('1hpv.pdb')
engine.set_binding_site((0.0, 0.0, 0.0), (20.0, 20.0, 20.0))
result = engine.dock_molecule('CCOc1ccc(cc1)C(C)C(=O)O', 'ibuprofen')
print('Ibuprofen docking score:', result['docking_score'])
"
```

### **Pipeline Validation**
```bash
# Validate complete pipeline with test molecules
python PureProt.py hybrid-screen batch_molecules.csv \
    --model egfr_model.joblib \
    --protein 1hpv.pdb \
    --center "0.0,0.0,0.0" \
    --test-mode
```

---

## 📊 **Results Analysis Commands**

### **Generate KICS Paper Results**
```bash
# Generate comprehensive benchmark results for publication
python generate_kics_results.py
```

### **Analyze Screening Results**
```bash
# Analyze and visualize screening results
python analyze_results.py screening_results.csv
```

### **Throughput Testing**
```bash
# Test system throughput and performance
python throughput_test.py
```

---

## 🛠 **Troubleshooting Commands**

### **Check Dependencies**
```bash
# Verify all required packages are installed
python -c "
import rdkit, sklearn, pandas, numpy, web3
print('All dependencies available')
"
```

### **Test Blockchain Connection**
```bash
# Test Purechain blockchain connectivity
python -c "
from workflow.verification_workflow import VerifiableDrugScreening
vds = VerifiableDrugScreening()
print('Blockchain connection:', 'OK' if vds.web3.is_connected() else 'Failed')
"
```

### **Memory Usage Check**
```bash
# Monitor memory usage during processing
python -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')
print(f'CPU cores: {psutil.cpu_count()}')
"
```

---

## 📈 **Production Deployment Commands**

### **Full Dataset Processing**
```bash
# Process complete natural products dataset (1,871 compounds)
python PureProt.py hybrid-screen natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --protein 1hpv.pdb \
    --center "0.0,0.0,0.0" \
    --size "20,20,20" \
    --output results_full_dataset.json \
    --log-level INFO
```

### **Continuous Monitoring**
```bash
# Run with detailed logging and monitoring
python PureProt.py hybrid-screen natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --protein 1hpv.pdb \
    --center "0.0,0.0,0.0" \
    --verbose \
    --save-intermediate \
    --checkpoint-interval 100
```

---

## 🎯 **Expected Outputs**

### **Screening Results Files**
- `screening_results.csv` - Tabular results with scores
- `screening_results.json` - Detailed JSON results
- `pureprot_results.json` - Complete pipeline results
- `blockchain_verification.json` - Blockchain transaction records

### **Performance Files**
- `model_performance_plot.png` - AI model performance visualization
- `throughput_results.json` - System performance metrics
- `paper_results.txt` - Summary for publication

### **Log Files**
- `pureprot.log` - Detailed execution logs
- `docking_results.log` - Docking-specific logs
- `blockchain_transactions.log` - Blockchain interaction logs

---

## ⏱️ **Estimated Execution Times**

| Command | Dataset Size | Estimated Time |
|---------|--------------|----------------|
| AI-only screening | 1,871 compounds | ~5 minutes |
| Docking-only | 1,871 compounds | ~7.6 hours |
| Hybrid screening | 1,871 compounds | ~8 hours |
| Batch processing | 100 compounds | ~25 minutes |
| Single molecule | 1 compound | ~15 seconds |

---

## 🚀 **Ready-to-Run Example**

```bash
# Complete end-to-end pipeline execution
echo "Starting PureProt hybrid screening pipeline..."

# Step 1: Verify system
python PureProt.py --version

# Step 2: Run hybrid screening
python PureProt.py hybrid-screen natural_products_for_screening.csv \
    --model egfr_model.joblib \
    --protein 1hpv.pdb \
    --center "0.0,0.0,0.0" \
    --size "20,20,20"

# Step 3: Generate results for KICS paper
python generate_kics_results.py

echo "Pipeline completed successfully!"
```

This complete command reference enables you to run the entire PureProt hybrid AI+docking+blockchain screening pipeline from start to finish!
