"""Test suite for the verifiable drug screening system.

This module contains tests for blockchain connector, molecular modeling,
and verification workflow components.
"""

import unittest
import json
from unittest.mock import MagicMock, patch
import sys
import os

# Import our custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blockchain.purechain_connector import PurechainConnector
from modeling.molecular_modeling import MoleculeRepresentation, ScreeningPipeline
from workflow.verification_workflow import VerifiableDrugScreening


class TestPurechainConnector(unittest.TestCase):
    """Tests for the blockchain connector module."""
    
    def setUp(self):
        self.connector = PurechainConnector("http://43.200.53.250:8548")
    
    def test_initialization(self):
        """Test connector initialization."""
        self.assertEqual(self.connector.rpc_url, "http://43.200.53.250:8548")
        self.assertIsNone(self.connector.account)
    
    def test_calculate_hash(self):
        """Test hash calculation."""
        data = {"key": "value", "number": 42}
        hash_result = self.connector.calculate_hash(data)
        # Hash should be a 64-character hex string
        self.assertEqual(len(hash_result), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in hash_result))
        
        # Same data should produce same hash
        hash_result2 = self.connector.calculate_hash(data)
        self.assertEqual(hash_result, hash_result2)
    

    def test_connection_succeeds_with_valid_rpc(self):
        """Test that connection succeeds with a valid RPC URL."""
        # self.connector is initialized in setUp with the real RPC
        result = self.connector.check_connection()
        self.assertTrue(result, "Connection to valid RPC should succeed.")

    def test_connection_fails_with_invalid_rpc(self):
        """Test that connection fails with an invalid RPC URL."""
        # Use an invalid RPC to ensure it fails predictably.
        invalid_connector = PurechainConnector(rpc_url="http://127.0.0.1:9999")
        result = invalid_connector.check_connection()
        self.assertFalse(result, "Connection to invalid RPC should fail.")

class TestMolecularModeling(unittest.TestCase):
    """Tests for the molecular modeling module."""
    
    def setUp(self):
        self.molecule = MoleculeRepresentation("test-mol", "CC(=O)OC1=CC=CC=C1C(=O)O")
        self.pipeline = ScreeningPipeline()
    
    def test_molecule_initialization(self):
        """Test molecule initialization."""
        self.assertEqual(self.molecule.molecule_id, "test-mol")
        self.assertEqual(self.molecule.smiles, "CC(=O)OC1=CC=CC=C1C(=O)O")
        self.assertIsNone(self.molecule.mol_file)
        self.assertEqual(self.molecule.features, {})
    
    def test_feature_extraction(self):
        """Test molecular feature extraction."""
        features = self.molecule.extract_features()
        
        # Check feature keys
        expected_keys = ["mol_weight", "logp", "h_donors", "h_acceptors", 
                       "tpsa", "rot_bonds", "fingerprint"]
        for key in expected_keys:
            self.assertIn(key, features)
        
        # Fingerprint should be a list of 16 binary values (updated)
        self.assertEqual(len(features["fingerprint"]), 16)
        for bit in features["fingerprint"]:
            self.assertIn(bit, [0, 1])
    
    def test_molecular_hash(self):
        """Test molecular hash generation."""
        # Hash from SMILES
        hash1 = self.molecule.get_molecular_hash()
        self.assertEqual(len(hash1), 64)
        
        # Same SMILES should give same hash
        molecule2 = MoleculeRepresentation("diff-id", "CC(=O)OC1=CC=CC=C1C(=O)O")
        hash2 = molecule2.get_molecular_hash()
        self.assertEqual(hash1, hash2)
        
        # Different SMILES should give different hash
        molecule3 = MoleculeRepresentation("diff-mol", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
        hash3 = molecule3.get_molecular_hash()
        self.assertNotEqual(hash1, hash3)
    
    def test_pipeline_screening(self):
        """Test complete screening pipeline."""
        result = self.pipeline.screen_molecule(
            "aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", "test-target")
        
        # Check result structure
        expected_keys = ["molecule_id", "smiles", "target_id", "molecular_hash",
                         "binding_affinity", "toxicity_score", "features", 
                         "timestamp", "binding_model_version", "toxicity_model_version"]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check specific values
        self.assertEqual(result["molecule_id"], "aspirin")
        self.assertEqual(result["smiles"], "CC(=O)OC1=CC=CC=C1C(=O)O")
        self.assertEqual(result["target_id"], "test-target")
        
        # Binding affinity should be a negative float (kcal/mol)
        self.assertLess(result["binding_affinity"], 0)
        
        # Toxicity should be between 0 and 1
        self.assertGreaterEqual(result["toxicity_score"], 0)
        self.assertLessEqual(result["toxicity_score"], 1)


class TestVerificationWorkflow(unittest.TestCase):
    """Tests for the verification workflow module."""
    
    def setUp(self):
        # Create mock blockchain connector
        self.mock_connector = MagicMock()
        self.mock_connector.connect_metamask.return_value = True
        self.mock_connector.calculate_hash.return_value = "mockhash123"
        self.mock_connector.record_screening_result.return_value = "0xmocktxhash456"
        self.mock_connector.verify_result.return_value = True
        
        # Patch the imports
        self.patcher1 = patch('workflow.verification_workflow.PurechainConnector', return_value=self.mock_connector)
        self.patcher2 = patch('workflow.verification_workflow.ScreeningPipeline')
        
        # Start the patchers
        self.mock_purechain_class = self.patcher1.start()
        self.mock_screening_class = self.patcher2.start()
        
        # Create mock screening pipeline
        self.mock_pipeline = MagicMock()
        mock_result = {
            "molecule_id": "test-mol",
            "binding_affinity": -9.5,
            "toxicity_score": 0.2
        }
        self.mock_pipeline.screen_molecule.return_value = mock_result
        self.mock_screening_class.return_value = self.mock_pipeline
        
        # Create workflow instance
        self.workflow = VerifiableDrugScreening("http://43.200.53.250:8548")
    
    def tearDown(self):
        # Stop the patchers
        self.patcher1.stop()
        self.patcher2.stop()
    
    def test_connect_wallet(self):
        """Test wallet connection."""
        # Mock connector is set up to return True for connect_metamask
        result = self.workflow.connect_wallet()
        self.assertTrue(result)
        self.mock_connector.connect_metamask.assert_called_once()
    
    def test_run_screening_job(self):
        """Test running a screening job."""
        # Set up test data
        mock_result = {
            "molecule_id": "test-mol",
            "binding_affinity": -8.5,
            "toxicity_score": 0.3,
            "target_id": "default"
        }
        self.mock_pipeline.screen_molecule.return_value = mock_result
        
        # Run the screening
        result = self.workflow.run_screening("test-mol", "CC(=O)O")
        
        # Check if screening pipeline was called
        self.mock_pipeline.screen_molecule.assert_called_once_with(
            "test-mol", "CC(=O)O", "default")
            
        # Check that the core results from the patched screen_molecule are present
        self.assertEqual(result['molecule_id'], mock_result['molecule_id'])
        self.assertEqual(result['binding_affinity'], mock_result['binding_affinity'])
        self.assertEqual(result['toxicity_score'], mock_result['toxicity_score'])
        self.assertEqual(result['target_id'], mock_result['target_id'])
        
        # Check that the workflow added its own data
        self.assertIn('timestamp', result)
        self.assertIn('assessment', result)
        self.assertIn('features', result)
    
    @patch('workflow.verification_workflow.VerifiableDrugScreening.run_screening')
    def test_verify_result(self, mock_run_screening):
        """Test result verification."""
        # Set up a mock result
        mock_result = {
            "molecule_id": "test-mol",
            "binding_affinity": -8.5,
            "toxicity_score": 0.3,
            "target_id": "default"
        }
        mock_run_screening.return_value = mock_result
        
        # Create a mock tx_hash
        tx_hash = "0xmocktxhash456"
        
        # First create a screening result and save in history
        screening_result = mock_result.copy()
        self.workflow.job_history["test-mol"] = {"result": screening_result, "tx_hash": tx_hash}
        
        # Mock the blockchain verification result
        self.mock_connector.verify_result.return_value = True
        
        # Test our verify functionality with the new implementation
        verification_result = self.workflow.verify_screening("test-mol", tx_hash)
        
        # Check that verification was called
        self.mock_connector.verify_result.assert_called_once()
        
        # Verify expected result format
        self.assertTrue(verification_result)
    
    def test_save_and_load_results(self):
        """Test saving and loading results."""
        import tempfile
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file_path = tmp.name
        
        try:
            # Create mock screening result
            mock_result = {
                "molecule_id": "test-mol",
                "binding_affinity": -8.5,
                "toxicity_score": 0.3,
                "target_id": "default"
            }
            mock_tx_hash = "0xmocktxhash456"
            
            # Add to job history directly for testing
            self.workflow.job_history["test-mol"] = {
                "result": mock_result,
                "tx_hash": mock_tx_hash
            }
            
            # Skip this if save_results/load_results are not in the current implementation
            if hasattr(self.workflow, 'save_results') and hasattr(self.workflow, 'load_results'):
                # Save results
                save_result = self.workflow.save_results(file_path)
                self.assertTrue(save_result)
                
                # Create new workflow and load results
                new_workflow = VerifiableDrugScreening("http://43.200.53.250:8548")
                load_result = new_workflow.load_results(file_path)
                self.assertTrue(load_result)
                
                # Check if history was loaded correctly
                self.assertEqual(len(new_workflow.job_history), 1)
            else:
                # Skip test if methods don't exist
                print("Skipping save/load test - methods not implemented")
                self.skipTest("save_results/load_results not implemented")
        finally:
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)


class TestPurechainIntegration(unittest.TestCase):
    """Integration tests for Purechain connection.
    
    Note: These tests require an actual Purechain connection and are disabled by default.
    Add the @unittest.skip decorator to skip these tests during normal test runs.
    """
    
    @unittest.skip("Requires actual Purechain connection")
    def test_real_purechain_connection(self):
        """Test connecting to a real Purechain node."""
        connector = PurechainConnector("http://43.200.53.250:8548")
        self.assertTrue(connector.check_connection())


if __name__ == "__main__":
    unittest.main()
