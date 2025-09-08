# PureChain Blockchain Integration Guide

## Overview

PureProt now fully integrates with the **PureChain blockchain network**, a zero-gas EVM-compatible blockchain designed for high-performance applications. This integration provides immutable verification of drug screening results with zero transaction costs.

## Key Features

### 🚀 Zero Gas Transactions
- **No transaction fees** - All blockchain operations are completely free
- **Instant finality** - Fast transaction confirmation
- **EVM compatibility** - Full Ethereum tooling support

### 🔒 Enhanced Security
- **Cryptographic verification** of screening results
- **Immutable audit trail** for drug discovery workflows
- **Tamper-proof data integrity** checks

### 🌐 Network Configuration

#### PureChain Testnet (Default)
- **Chain ID**: `900520900520`
- **RPC URL**: `https://purechainnode.com:8547`
- **Gas Price**: `0` (Zero fees)
- **Current Block**: 78,000+ and growing

#### PureChain Mainnet
- **Chain ID**: `900520900520`
- **RPC URL**: `https://purechainnode.com:8547`
- **Gas Price**: `0` (Zero fees)

## Enhanced PurechainConnector Features

### Automatic Network Detection
The enhanced `PurechainConnector` automatically configures optimal settings:

```python
from blockchain.purechain_connector import PurechainConnector

# Use PureChain testnet (default)
connector = PurechainConnector(network='testnet')

# Use PureChain mainnet
connector = PurechainConnector(network='mainnet')

# Use local development (Ganache)
connector = PurechainConnector(network='local')
```

### Zero-Gas Transaction Optimization
- Automatic gas price detection (0 for PureChain networks)
- Enhanced logging for transaction monitoring
- Optimized transaction parameters for PureChain

### Network-Aware Configuration
```python
# The connector automatically detects PureChain networks
if connector.is_purechain:
    print("Using PureChain network with zero gas fees!")
    print(f"Chain ID: {connector.chain_id}")
    print(f"RPC: {connector.rpc_url}")
```

## Integration with Hybrid Screening

### Enhanced VerifiableDrugScreening
The workflow now supports flexible network configuration:

```python
from workflow.verification_workflow import VerifiableDrugScreening

# Use PureChain testnet (recommended)
screening = VerifiableDrugScreening(network='testnet')

# Use PureChain mainnet for production
screening = VerifiableDrugScreening(network='mainnet')

# Custom network configuration
screening = VerifiableDrugScreening(
    rpc_url='https://custom-node.com:8545',
    chain_id=12345,
    network='custom'
)
```

### Hybrid Screening with Blockchain Recording
```python
# Run hybrid screening with blockchain verification
result = screening.run_screening_job(
    molecule_id="compound_001",
    smiles="CCO",  # Ethanol
    target_id="1HPV",
    custom_data={
        "docking_score": -8.5,
        "ai_prediction": 0.85,
        "consensus_score": 0.78,
        "drug_likeness": {
            "lipinski_violations": 0,
            "molecular_weight": 46.07
        }
    }
)

print(f"Blockchain transaction: {result['blockchain_result']['tx_hash']}")
print(f"Gas used: {result['blockchain_result']['gas_used']} (Cost: $0.00)")
```

## CLI Integration

### Using PureChain with PureProt CLI
```bash
# Run hybrid screening with PureChain blockchain recording
python PureProt.py hybrid-screen \
    --smiles "CCO" \
    --molecule-id "compound_001" \
    --protein "1hpv_prepared.pdb" \
    --blockchain \
    --network testnet

# Verify results on blockchain
python PureProt.py verify-result \
    --tx-hash "0x123abc..." \
    --network testnet
```

## Environment Configuration

### Required Environment Variables
Create a `.env` file in your project root:

```env
# PureChain wallet private key (for signing transactions)
TEST_PRIVATE_KEY=0x1234567890abcdef...

# Optional: Custom RPC endpoint
PURECHAIN_RPC_URL=https://purechainnode.com:8547

# Optional: Custom contract address
CONTRACT_ADDRESS=0xabcdef1234567890...
```

### Contract Deployment Information
The system loads contract information from `local_deployment_info.json`:

```json
{
  "address": "0x1234567890abcdef1234567890abcdef12345678",
  "abi": [...],
  "network": "testnet",
  "chain_id": 900520900520
}
```

## Advanced Features

### Performance Monitoring
The enhanced connector provides detailed transaction metrics:

```python
result = connector.record_and_verify_result(
    result_hash_bytes=b"...",
    molecule_data_hash_bytes=b"...",
    molecule_id="compound_001"
)

print(f"Transaction time: {result['duration']:.2f}s")
print(f"Block number: {result['block_number']}")
print(f"Gas used: {result['gas_used']}")
print(f"Transaction cost: $0.00 (Zero gas fees!)")
```

### Enhanced Verification
```python
# Verify result integrity
verification = connector.verify_result(
    tx_hash="0x123abc...",
    original_result_hash=result_hash,
    original_molecule_hash=molecule_hash
)

if verification['verified']:
    print("✅ Result verified on blockchain!")
    print(f"Verification time: {verification['duration']:.2f}s")
else:
    print("❌ Verification failed!")
    print(f"Reason: {verification['reason']}")
```

## Benefits of PureChain Integration

### 🆓 Cost Efficiency
- **Zero transaction fees** eliminate blockchain costs
- **No gas optimization** needed - all transactions are free
- **Unlimited transactions** for large-scale screening

### ⚡ Performance
- **Fast confirmation times** for immediate verification
- **High throughput** for batch processing
- **No network congestion** issues

### 🔒 Security & Compliance
- **Immutable audit trails** for regulatory compliance
- **Cryptographic proof** of result integrity
- **Decentralized verification** without single points of failure

### 🌐 Interoperability
- **EVM compatibility** with existing Ethereum tools
- **Web3.py integration** for familiar development experience
- **Standard smart contract** interfaces

## Troubleshooting

### Common Issues

#### Connection Problems
```python
# Test connection
connector = PurechainConnector(network='testnet')
if connector.w3.is_connected():
    print("✅ Connected to PureChain!")
else:
    print("❌ Connection failed - check network settings")
```

#### Private Key Issues
```bash
# Ensure private key is in .env file
echo "TEST_PRIVATE_KEY=0x..." >> .env

# Verify key format (should start with 0x)
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('TEST_PRIVATE_KEY'))"
```

#### Contract Address Issues
```python
# Verify contract deployment
with open('local_deployment_info.json', 'r') as f:
    info = json.load(f)
    print(f"Contract: {info['address']}")
    print(f"Network: {info.get('network', 'unknown')}")
```

## Migration from Legacy Setup

### Updating Existing Code
```python
# Old way (manual configuration)
connector = PurechainConnector(
    rpc_url="http://127.0.0.1:8545",
    chain_id=1337,
    private_key=private_key,
    contract_address=contract_address
)

# New way (automatic PureChain configuration)
connector = PurechainConnector(
    network='testnet',  # Uses PureChain testnet automatically
    private_key=private_key,
    contract_address=contract_address
)
```

### Benefits of Migration
- **Zero gas costs** instead of ETH/test token fees
- **Faster transactions** with immediate finality
- **Better reliability** with dedicated PureChain infrastructure
- **Enhanced logging** and monitoring capabilities

## Next Steps

1. **Test Integration**: Run hybrid screening with PureChain testnet
2. **Deploy to Production**: Switch to PureChain mainnet for production use
3. **Monitor Performance**: Use built-in metrics for optimization
4. **Scale Operations**: Leverage zero-gas costs for large-scale screening

## Support and Resources

- **PureChain Documentation**: [Official PureChain Docs]
- **PureChainLib SDK**: https://www.npmjs.com/package/purechainlib
- **Network Status**: Monitor at https://purechainnode.com:8547
- **Community Support**: [PureChain Discord/Telegram]

---

*This integration guide covers the enhanced PureChain blockchain integration in PureProt v2.0, providing zero-cost, high-performance blockchain verification for drug discovery workflows.*
