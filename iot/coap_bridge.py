#!/usr/bin/env python3
"""
PureProtX CoAP Bridge (Stub)

CoAP (Constrained Application Protocol, RFC 7252) is the request/response
counterpart to MQTT for extremely resource-constrained IoT devices.
This module provides a CoAP endpoint for the PureProtX pipeline,
suitable for devices that cannot maintain persistent TCP connections.

Note: Full CoAP implementation requires the `aiocoap` library.
This module provides the interface and falls back to direct pipeline
invocation when CoAP infrastructure is unavailable.

CoAP resources:
  POST /pureprot/screen    -- submit compounds for screening
  GET  /pureprot/result    -- retrieve results by job_id
  GET  /pureprot/audit     -- retrieve audit digest by job_id
"""

import os
import sys
import json
import hashlib
import time
import logging
from typing import Dict, Any, Optional, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


def _coap_available() -> bool:
    """Check if aiocoap is available."""
    try:
        import aiocoap
        return True
    except ImportError:
        return False


def handle_screen_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a CoAP POST /pureprot/screen request.

    Payload format (same as MQTT ingest):
      {
        "job_id": "string",
        "target": "CHEMBL243",
        "smiles": ["CCO", ...],
        "requester_node": "edge-node-01"
      }

    Returns result dict with ranked hits and digest.
    """
    from iot.mqtt_bridge import _run_screening_pipeline, _commit_digest_to_purechain

    job_id = payload.get("job_id", f"coap_{int(time.time())}")
    target = payload.get("target", "CHEMBL243")
    smiles_list = payload.get("smiles", [])

    t0 = time.perf_counter()
    pipeline_result = _run_screening_pipeline(smiles_list, target)
    bc_result = _commit_digest_to_purechain(pipeline_result["digest"], job_id)
    e2e_ms = (time.perf_counter() - t0) * 1000

    block_number = bc_result.get("block_number", 0) if bc_result.get("success") else 0

    return {
        "job_id": job_id,
        "target": target,
        "ranked_hits": pipeline_result["ranked_hits"][:20],
        "digest": pipeline_result["digest"],
        "block_number": block_number,
        "processing_time_ms": round(e2e_ms, 1),
        "protocol": "coap",
    }


def start_coap_server(bind_address: str = "::", port: int = 5683) -> None:
    """
    Start the CoAP server.  Requires aiocoap.

    Raises ImportError if aiocoap is not installed.
    """
    if not _coap_available():
        raise ImportError(
            "aiocoap is required for CoAP server. "
            "Install with: pip install aiocoap"
        )

    import asyncio
    import aiocoap
    import aiocoap.resource as resource

    class ScreenResource(resource.Resource):
        async def render_post(self, request):
            try:
                payload = json.loads(request.payload.decode())
                result = handle_screen_request(payload)
                return aiocoap.Message(
                    code=aiocoap.CONTENT,
                    payload=json.dumps(result).encode(),
                )
            except Exception as e:
                return aiocoap.Message(
                    code=aiocoap.INTERNAL_SERVER_ERROR,
                    payload=str(e).encode(),
                )

    root = resource.Site()
    root.add_resource(["pureprot", "screen"], ScreenResource())

    asyncio.get_event_loop().run_until_complete(
        aiocoap.Context.create_server_context(root, bind=(bind_address, port))
    )
    logger.info("CoAP server listening on [%s]:%d", bind_address, port)
    asyncio.get_event_loop().run_forever()


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 60)
    print("PureProtX CoAP Bridge")
    print("=" * 60)

    if _coap_available():
        start_coap_server()
    else:
        print("aiocoap not installed. Running direct pipeline test instead.")
        test_payload = {
            "job_id": "coap_test_001",
            "target": "CHEMBL243",
            "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
            "requester_node": "test-node",
        }
        result = handle_screen_request(test_payload)
        print(json.dumps(result, indent=2))
