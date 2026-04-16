#!/usr/bin/env python3
"""
PureProtX MQTT Ingestion Bridge

Allows IoT devices to submit SMILES strings and target IDs to the PureProtX
pipeline via MQTT (the standard IoT pub/sub protocol).  The bridge runs the
deterministic pipeline and publishes ranked hits and SHA-256 audit digests
back over MQTT, while also committing digests to PureChain.

MQTT topics:
  Subscribe:        pureprot/compounds/ingest
  Publish results:  pureprot/compounds/results
  Publish audit:    pureprot/audit/hashes
  Status:           pureprot/system/status

Dependencies: paho-mqtt>=2.0.0

Output:
  results/mqtt_benchmark.csv
"""

import os
import sys
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# MQTT topic constants
TOPIC_INGEST = "pureprot/compounds/ingest"
TOPIC_RESULTS = "pureprot/compounds/results"
TOPIC_AUDIT = "pureprot/audit/hashes"
TOPIC_STATUS = "pureprot/system/status"


# ------------------------------------------------------------------
# Pipeline integration
# ------------------------------------------------------------------

# In-process model cache -- loading a joblib model takes ~2s, so we cache
# it across MQTT messages to realistic production performance.
_MODEL_CACHE: Dict[str, Any] = {}


def _get_cached_model(target: str):
    """Load and cache the ConsensusAIModel for a target."""
    if target in _MODEL_CACHE:
        return _MODEL_CACHE[target]

    try:
        from pureprot.ai_model import ConsensusAIModel
        model = ConsensusAIModel()
        model_path = os.path.join(
            PROJECT_ROOT, "experiments", "paper_results", "models",
            f"{target}_model.joblib",
        )
        if os.path.exists(model_path):
            model.load_model(model_path)
            _MODEL_CACHE[target] = model
            return model
    except Exception as e:
        logger.warning("Model load failed for %s: %s", target, e)

    _MODEL_CACHE[target] = None
    return None


def _run_screening_pipeline(smiles_list: List[str], target: str) -> Dict[str, Any]:
    """
    Run the PureProtX screening pipeline on a batch of SMILES.

    Returns dict with ranked_hits and SHA-256 digest.
    """
    t0 = time.perf_counter()

    # Compute canonical JSON of input for hashing
    canonical_input = json.dumps(
        {"target": target, "smiles": sorted(smiles_list)},
        sort_keys=True, separators=(",", ":"),
    )
    input_digest = hashlib.sha256(canonical_input.encode()).hexdigest()

    # Try to use ConsensusAIModel for scoring (cached across calls)
    ranked_hits = []
    try:
        from pureprot.ai_model import ConsensusAIModel

        model = _get_cached_model(target)
        if model is not None and model.is_trained:
            for i, smi in enumerate(smiles_list):
                try:
                    pred = model.predict_single(smi)
                    ranked_hits.append({
                        "smiles": smi,
                        "score": round(pred["consensus"], 4),
                        "rank": 0,  # filled after sorting
                    })
                except Exception:
                    ranked_hits.append({
                        "smiles": smi,
                        "score": 0.0,
                        "rank": 0,
                    })
        else:
            # No model available -- use placeholder scores
            for smi in smiles_list:
                ranked_hits.append({"smiles": smi, "score": 0.0, "rank": 0})
    except ImportError:
        for smi in smiles_list:
            ranked_hits.append({"smiles": smi, "score": 0.0, "rank": 0})

    # Sort by score descending, assign ranks
    ranked_hits.sort(key=lambda x: x["score"], reverse=True)
    for rank, hit in enumerate(ranked_hits, 1):
        hit["rank"] = rank

    # Compute result digest
    result_json = json.dumps(ranked_hits, sort_keys=True, separators=(",", ":"))
    result_digest = hashlib.sha256(result_json.encode()).hexdigest()

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "ranked_hits": ranked_hits,
        "digest": result_digest,
        "input_digest": input_digest,
        "processing_time_ms": round(elapsed_ms, 1),
    }


_BC_CONNECTOR = None
_BC_INIT_LOCK = __import__("threading").Lock()


def _commit_digest_to_purechain(digest_hex: str, job_id: str) -> Dict[str, Any]:
    """Commit a result digest to PureChain. Returns tx info or None.

    Reuses a single connector across calls to avoid reconnecting on every
    MQTT message (each connect costs ~600 ms of HTTP round-trips).
    Thread-safe initialization via _BC_INIT_LOCK.
    """
    global _BC_CONNECTOR
    if _BC_CONNECTOR is None:
        with _BC_INIT_LOCK:
            if _BC_CONNECTOR is None:  # double-check after acquiring lock
                from blockchain.purechain_factory import get_purechain_connector
                _BC_CONNECTOR = get_purechain_connector(strict=False)

    if _BC_CONNECTOR is None:
        return {"success": False, "offline": True}

    try:
        result_hash = bytes.fromhex(digest_hex)
        data_hash = hashlib.sha256(job_id.encode()).digest()
        return _BC_CONNECTOR.record_and_verify_result(result_hash, data_hash, job_id)
    except Exception as e:
        logger.warning("PureChain commit failed: %s", e)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# MQTT Bridge
# ------------------------------------------------------------------

def _mqtt_available() -> bool:
    """Check if paho-mqtt is importable."""
    try:
        import paho.mqtt.client as mqtt
        return True
    except ImportError:
        return False


def start_bridge(
    broker_host: str = "localhost",
    broker_port: int = 1883,
    pipeline_config: str = "config/default.yaml",
    max_workers: int = 4,
    async_mode: bool = True,
) -> None:
    """Starts the MQTT bridge. Blocking call.

    Args:
        max_workers: Max concurrent pipeline threads (async mode only).
        async_mode:  If True, process messages concurrently via ThreadPoolExecutor.
                     If False, process sequentially (original behaviour).
    """
    import paho.mqtt.client as mqtt
    from concurrent.futures import ThreadPoolExecutor
    import threading

    executor = ThreadPoolExecutor(max_workers=max_workers) if async_mode else None
    publish_lock = threading.Lock()

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info("Connected to MQTT broker at %s:%d", broker_host, broker_port)
            client.subscribe(TOPIC_INGEST)
            client.publish(TOPIC_STATUS, json.dumps({
                "status": "online",
                "mode": "async" if async_mode else "sequential",
                "max_workers": max_workers if async_mode else 1,
                "timestamp": time.time(),
            }))
        else:
            logger.error("MQTT connection failed with code %d", rc)

    def on_message(client, userdata, msg):
        if executor is not None:
            executor.submit(_process_message, client, msg, publish_lock)
        else:
            _process_message(client, msg, publish_lock)

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="pureprot-bridge",
    )
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker_host, broker_port, keepalive=60)
    mode_str = f"async (max_workers={max_workers})" if async_mode else "sequential"
    logger.info("Starting MQTT bridge [%s] (listening on %s)", mode_str, TOPIC_INGEST)
    client.loop_forever()


def _process_message(client, msg, publish_lock) -> None:
    """Process a single ingest message (runs in a worker thread when async)."""
    t0 = time.perf_counter()

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("Invalid ingest message: %s", e)
        return

    job_id = payload.get("job_id", f"job_{int(time.time())}")
    target = payload.get("target", "CHEMBL243")
    smiles_list = payload.get("smiles", [])
    requester = payload.get("requester_node", "unknown")

    logger.info("Ingest job %s: %d compounds for %s from %s",
                job_id, len(smiles_list), target, requester)

    # Run pipeline
    pipeline_result = _run_screening_pipeline(smiles_list, target)

    # Commit to PureChain
    bc_result = _commit_digest_to_purechain(pipeline_result["digest"], job_id)
    block_number = bc_result.get("block_number", 0) if bc_result.get("success") else 0

    e2e_ms = (time.perf_counter() - t0) * 1000

    # Publish results (thread-safe)
    result_msg = {
        "job_id": job_id,
        "target": target,
        "ranked_hits": pipeline_result["ranked_hits"][:20],
        "digest": pipeline_result["digest"],
        "block_number": block_number,
        "processing_time_ms": round(e2e_ms, 1),
    }

    with publish_lock:
        if client is not None and hasattr(client, "publish"):
            client.publish(TOPIC_RESULTS, json.dumps(result_msg))
            client.publish(TOPIC_AUDIT, json.dumps({
                "job_id": job_id,
                "digest": pipeline_result["digest"],
                "block_number": block_number,
                "timestamp": time.time(),
            }))

    logger.info("Job %s completed in %.1f ms (block %s)",
                job_id, e2e_ms, block_number or "offline")


# ------------------------------------------------------------------
# Throughput benchmark
# ------------------------------------------------------------------

def benchmark_mqtt_throughput(
    n_messages: int = 100,
    compounds_per_message: int = 50,
    broker_host: str = "localhost",
    broker_port: int = 1883,
) -> Dict[str, Any]:
    """
    Publishes n_messages batches and measures:
    - Message ingestion rate (msg/s)
    - End-to-end latency per batch (ms)
    - Blockchain commit success rate (%)

    Returns results dict, saves to results/mqtt_benchmark.csv
    """
    use_mqtt = _mqtt_available()

    # Generate synthetic compound batches
    rng = np.random.RandomState(42)
    sample_smiles = [
        "CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "C1CCCCC1", "C1=CC=CC=C1O", "CCN(CC)CC", "CCCCCCCC",
    ]

    results_rows = []
    bc_successes = 0

    if use_mqtt:
        import paho.mqtt.client as mqtt

        # Set up subscriber to catch results
        received = []

        def on_result(client, userdata, msg):
            # Stamp receive time so end-to-end latency reflects
            # publish->broker->bridge->pipeline->broker->subscriber.
            r = json.loads(msg.payload.decode())
            r["_receive_perf"] = time.perf_counter()
            received.append(r)

        sub_client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="pureprot-bench-sub",
        )
        pub_client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="pureprot-bench-pub",
        )

        try:
            sub_client.on_message = on_result
            sub_client.connect(broker_host, broker_port)
            sub_client.subscribe(TOPIC_RESULTS)
            sub_client.loop_start()

            pub_client.connect(broker_host, broker_port)

            t_start = time.perf_counter()

            # Track per-message publish time to measure real end-to-end latency
            publish_times = {}
            for i in range(n_messages):
                smiles_batch = list(rng.choice(sample_smiles, compounds_per_message, replace=True))
                payload = {
                    "job_id": f"bench_{i}",
                    "target": "CHEMBL243",
                    "smiles": smiles_batch,
                    "requester_node": "benchmark-client",
                }
                publish_times[f"bench_{i}"] = time.perf_counter()
                pub_client.publish(TOPIC_INGEST, json.dumps(payload))

            # Wait until either every message is back OR we exceed a generous
            # timeout.  Each message takes ~2.5 s (pipeline + blockchain) so
            # we budget 10 s per message plus 15 s slack.
            deadline = time.perf_counter() + n_messages * 10 + 15
            while len(received) < n_messages and time.perf_counter() < deadline:
                time.sleep(0.2)
            total_time = time.perf_counter() - t_start

            # Compute per-message end-to-end latencies from publish->receive timestamps
            for r in received:
                jid = r.get("job_id")
                if jid in publish_times and "_receive_perf" in r:
                    r["_e2e_ms"] = (r["_receive_perf"] - publish_times[jid]) * 1000

            sub_client.loop_stop()
            sub_client.disconnect()
            pub_client.disconnect()

            msg_rate = n_messages / total_time if total_time > 0 else 0
            for r in received:
                # Prefer the real broker-measured E2E latency over the
                # pipeline's self-reported processing_time_ms
                latency = r.get("_e2e_ms", r.get("processing_time_ms", 0))
                pipeline_ms = r.get("processing_time_ms", 0)
                bc_ok = r.get("block_number", 0) > 0
                if bc_ok:
                    bc_successes += 1
                results_rows.append({
                    "job_id": r.get("job_id"),
                    "latency_ms": round(latency, 1),
                    "pipeline_ms": round(pipeline_ms, 1),
                    "blockchain_committed": bc_ok,
                    "block_number": r.get("block_number", 0),
                })

        except Exception as e:
            logger.warning("MQTT benchmark via broker failed: %s -- falling back to offline", e)
            use_mqtt = False

    if not use_mqtt:
        # Offline benchmark: bypass MQTT, call pipeline directly
        print("  [mqtt_benchmark] Running in OFFLINE mode (no MQTT broker)")
        t_start = time.perf_counter()

        for i in range(n_messages):
            smiles_batch = list(rng.choice(sample_smiles, compounds_per_message, replace=True))

            t_msg = time.perf_counter()
            pipeline_result = _run_screening_pipeline(smiles_batch, "CHEMBL243")
            bc_result = _commit_digest_to_purechain(pipeline_result["digest"], f"bench_{i}")
            latency_ms = (time.perf_counter() - t_msg) * 1000

            bc_ok = bc_result.get("success", False) if bc_result else False
            if bc_ok:
                bc_successes += 1

            results_rows.append({
                "job_id": f"bench_{i}",
                "latency_ms": round(latency_ms, 1),
                "blockchain_committed": bc_ok,
            })

        total_time = time.perf_counter() - t_start
        msg_rate = n_messages / total_time if total_time > 0 else 0

    df = pd.DataFrame(results_rows)
    mean_latency = float(df["latency_ms"].mean()) if not df.empty else 0.0
    bc_rate = (bc_successes / n_messages) * 100 if n_messages > 0 else 0.0

    summary = {
        "n_messages": n_messages,
        "compounds_per_message": compounds_per_message,
        "total_time_s": round(total_time, 2),
        "message_rate_per_s": round(msg_rate, 2),
        "mean_latency_ms": round(mean_latency, 1),
        "blockchain_commit_success_pct": round(bc_rate, 1),
        "offline_mode": not use_mqtt,
    }

    # Save CSV
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "mqtt_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    # Print summary
    print(f"  Messages: {n_messages}, Rate: {msg_rate:.1f} msg/s, "
          f"Latency: {mean_latency:.0f} ms, BC: {bc_rate:.0f}%")

    return summary


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="PureProtX MQTT Bridge")
    parser.add_argument("--mode", choices=["bridge", "benchmark"], default="benchmark")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--n-messages", type=int, default=100)
    parser.add_argument("--sync", action="store_true",
                        help="Run bridge in sequential mode (default: async)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max concurrent workers in async mode")
    args = parser.parse_args()

    if args.mode == "bridge":
        print("=" * 60)
        print("PureProtX MQTT Bridge")
        print("=" * 60)
        start_bridge(
            broker_host=args.broker_host, broker_port=args.broker_port,
            async_mode=not args.sync, max_workers=args.workers,
        )
    else:
        print("=" * 60)
        print("PureProtX MQTT Throughput Benchmark")
        print("=" * 60)
        result = benchmark_mqtt_throughput(
            n_messages=args.n_messages,
            broker_host=args.broker_host,
            broker_port=args.broker_port,
        )
        print(json.dumps(result, indent=2))
