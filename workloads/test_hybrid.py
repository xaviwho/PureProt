#!/usr/bin/env python3
"""Test script for hybrid screening functionality"""

import sys
import os
sys.path.insert(0, '.')

def test_docking_engine():
    """Test the DockingEngine functionality"""
    print("=== Testing DockingEngine ===")
    
    try:
        from modeling.docking_engine import DockingEngine
        
        # Initialize docking engine
        docking = DockingEngine()
        print("✓ DockingEngine initialized successfully")
        
        # Test molecule docking
        test_smiles = "CCOc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen-like molecule
        result = docking.dock_molecule(test_smiles, "test_molecule")
        
        print(f"✓ Docking completed")
        print(f"  - Molecule ID: {result.get('molecule_id')}")
        print(f"  - Docking Score: {result.get('docking_score')}")
        print(f"  - Status: {result.get('status')}")
        print(f"  - Method: {result.get('method')}")
        
        return True
        
    except Exception as e:
        print(f"✗ DockingEngine test failed: {e}")
        return False

def test_hybrid_screening():
    """Test the HybridScreening functionality"""
    print("\n=== Testing HybridScreening ===")
    
    try:
        from modeling.docking_engine import DockingEngine, HybridScreening
        
        # Initialize components
        docking_engine = DockingEngine()
        hybrid = HybridScreening(docking_engine=docking_engine)
        print("✓ HybridScreening initialized successfully")
        
        # Test hybrid screening
        test_smiles = "CC(=O)NC1=CC=C(C=C1)O"  # Paracetamol
        result = hybrid.hybrid_screen("paracetamol", test_smiles)
        
        print(f"✓ Hybrid screening completed")
        print(f"  - Molecule ID: {result.get('molecule_id')}")
        print(f"  - Docking Score: {result.get('docking_score')}")
        print(f"  - Consensus Score: {result.get('consensus_score')}")
        print(f"  - Drug-like: {result.get('drug_like')}")
        
        return True
        
    except Exception as e:
        print(f"✗ HybridScreening test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing with existing molecules"""
    print("\n=== Testing Batch Processing ===")
    
    try:
        import csv
        from modeling.docking_engine import DockingEngine
        
        # Read existing batch_molecules.csv
        molecules = []
        with open('batch_molecules.csv', 'r') as f:
            reader = csv.DictReader(f)
            molecules = list(reader)
        
        print(f"✓ Loaded {len(molecules)} molecules from batch_molecules.csv")
        
        # Initialize docking engine
        docking = DockingEngine()
        
        # Process molecules
        results = docking.batch_dock(molecules)
        
        print(f"✓ Batch docking completed")
        print(f"  - Processed: {len(results)} molecules")
        
        for result in results:
            print(f"  - {result['molecule_id']}: {result['docking_score']} ({result['status']})")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False

if __name__ == "__main__":
    print("PureProt Hybrid Screening Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_docking_engine():
        tests_passed += 1
    
    if test_hybrid_screening():
        tests_passed += 1
    
    if test_batch_processing():
        tests_passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Hybrid screening is ready for KICS paper.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
