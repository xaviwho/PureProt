"""
Shared PureChain connector factory.

Reads .env to determine which network to connect to (mainnet by default).
Returns a fully configured PurechainConnector or None on failure.
All IoT/edge modules use this so the connector setup lives in one place.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_purechain_connector(strict: bool = False):
    """
    Build a PurechainConnector from environment variables.

    Honours .env settings:
      PURECHAIN_RPC_URL  -- node URL (default: PureChain mainnet)
      CONTRACT_ADDRESS   -- deployed verifier address
      TEST_PRIVATE_KEY   -- wallet for signing transactions
      NETWORK            -- 'mainnet' | 'testnet' | 'local'  (default: mainnet)

    For local Ganache development, set:
      NETWORK=local  and  PURECHAIN_RPC_URL=http://127.0.0.1:8545

    If strict=True, raises on failure; otherwise returns None.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from blockchain.purechain_connector import PurechainConnector

    rpc_url = os.getenv("PURECHAIN_RPC_URL", "https://purechainnode.com")
    network = os.getenv("NETWORK", "mainnet").lower()
    private_key = os.getenv("TEST_PRIVATE_KEY")
    contract_address = os.getenv("CONTRACT_ADDRESS")

    # Fallback to local_deployment_info.json for contract address
    if not contract_address:
        info_path = PROJECT_ROOT / "local_deployment_info.json"
        if info_path.exists():
            with open(info_path) as f:
                contract_address = json.load(f).get("contract_address")

    if not contract_address:
        msg = "No CONTRACT_ADDRESS in .env or local_deployment_info.json"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
        return None

    # Resolve chain_id from network
    if network in ("mainnet", "testnet"):
        chain_id = 900520900520  # PureChain
    else:
        chain_id = int(os.getenv("CHAIN_ID", "1337"))

    try:
        return PurechainConnector(
            rpc_url=rpc_url,
            contract_address=contract_address,
            private_key=private_key,
            chain_id=chain_id,
            network=network,
        )
    except Exception as e:
        if strict:
            raise
        logger.warning("PureChain connection failed: %s", e)
        return None
