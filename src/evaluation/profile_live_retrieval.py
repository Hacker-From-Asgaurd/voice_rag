import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from harness.guardrails import calibrate_crossencoder_score

PROFILE_OUTPUT_FILE = Path("data/live_retrieval_latency_profile.json")
OPTIMIZED_OUTPUT_FILE = Path("data/e5_base_optimized_latency.json")

TEST_QUERIES = [
    "मैनहट्टन परियोजना क्या थी?",
    "What was the purpose of the Manhattan Project?",
    "भारत के पहले प्रधानमंत्री कौन थे?",
    "What is the freezing point of water?",
    "मॅनहॅटन प्रकल्प काय होता?",
    "who discovered gravity?",
    "द्वितीय विश्व युद्ध कब समाप्त हुआ?",
    "what is photosynthesis process in plants?",
    "ताज महल कहाँ स्थित है?",
    "who is the author of mahabharata?"
]

def profile_pipeline():
    print("\n=======================================================")
    print("      LIVE SINGLE-QUERY RETRIEVAL CORE PROFILER        ")
    print("=======================================================\n")

    # 1. Initialize models ONCE
    t_init_start = time.perf_counter()
    retriever = Retriever()
    reranker = Reranker()
    init_time_ms = (time.perf_counter() - t_init_start) * 1000.0
    print(f"Model initialization completed in {init_time_ms:.1f} ms\n")

    has_cuda = torch.cuda.is_available()

    def sync_gpu():
        if has_cuda:
            torch.cuda.synchronize()

    def profile_single_query(query: str, top_k: int = 15, rerank_top_k: int = 5) -> Dict[str, float]:
        timings = {}
        t_total_start = time.perf_counter()

        # Step 1: Query prefix & tokenization
        t0 = time.perf_counter()
        prefixed_query = "query: " + query
        timings["1_tokenization_prep_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 2: E5 forward pass
        t0 = time.perf_counter()
        with torch.inference_mode():
            query_emb = retriever.model.encode([prefixed_query], normalize_embeddings=True, show_progress_bar=False)
        sync_gpu()
        timings["2_e5_forward_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 3: Embedding conversion to float32
        t0 = time.perf_counter()
        query_emb_np = np.asarray(query_emb, dtype="float32")
        timings["3_emb_conversion_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 4: FAISS search
        t0 = time.perf_counter()
        scores, indices = retriever.index.search(query_emb_np, top_k)
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                rec = retriever.metadata[idx]
                candidates.append({
                    "score": float(score),
                    "query_id": rec["query_id"],
                    "passage_id": rec["passage_id"],
                    "is_selected": rec["is_selected"],
                    "chunk": rec["chunk"],
                    "parent": rec["parent"]
                })
        timings["4_faiss_search_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 5: CrossEncoder preprocessing
        t0 = time.perf_counter()
        pairs = [(query, c["chunk"]) for c in candidates]
        timings["5_crossencoder_prep_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 6: CrossEncoder inference
        t0 = time.perf_counter()
        if pairs:
            with torch.inference_mode():
                ce_scores = reranker.model.predict(pairs, show_progress_bar=False)
            sync_gpu()
            for c, s in zip(candidates, ce_scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            reranked = candidates[:rerank_top_k]
        else:
            reranked = []
        timings["6_crossencoder_inference_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 7: Evidence gate
        t0 = time.perf_counter()
        evidence = []
        for r in reranked:
            raw_s = float(r.get("rerank_score", 0.0))
            calib = calibrate_crossencoder_score(raw_s)
            r["calibrated_confidence"] = calib
            if calib >= 0.80 or raw_s >= 0.80:
                evidence.append(r)
        timings["7_evidence_gate_ms"] = (time.perf_counter() - t0) * 1000.0

        timings["total_retrieval_core_ms"] = (time.perf_counter() - t_total_start) * 1000.0
        return timings

    # ---------------------------------------------------------
    # 2. Cold Latency (First run before warmup)
    # ---------------------------------------------------------
    print("Measuring COLD latency (Run #1)...")
    cold_profile = profile_single_query("मैनहट्टन परियोजना क्या थी?")
    print(f"Cold Total Retrieval Core: {cold_profile['total_retrieval_core_ms']:.1f} ms "
          f"(E5: {cold_profile['2_e5_forward_ms']:.1f} ms, CE: {cold_profile['6_crossencoder_inference_ms']:.1f} ms, FAISS: {cold_profile['4_faiss_search_ms']:.1f} ms)\n")

    # ---------------------------------------------------------
    # 3. Warm-up Iterations (10 queries)
    # ---------------------------------------------------------
    print("Performing 10 warm-up requests...")
    for q in TEST_QUERIES:
        _ = profile_single_query(q)
    print("Warm-up complete.\n")

    # ---------------------------------------------------------
    # 4. Measured Warm Requests (30 queries: 3 rounds x 10 queries)
    # ---------------------------------------------------------
    print("Executing 30 measured warm queries across diverse test inputs...")
    warm_measurements: List[Dict[str, float]] = []

    for round_idx in range(3):
        for q in TEST_QUERIES:
            t_data = profile_single_query(q)
            warm_measurements.append(t_data)

    # ---------------------------------------------------------
    # 5. Compute Statistics
    # ---------------------------------------------------------
    keys = list(warm_measurements[0].keys())
    stats = {}

    for k in keys:
        vals = sorted([m[k] for m in warm_measurements])
        n = len(vals)
        stats[k] = {
            "mean_ms": round(float(np.mean(vals)), 2),
            "p50_ms": round(float(vals[int(n * 0.50)]), 2),
            "p70_ms": round(float(vals[int(n * 0.70)]), 2),
            "p95_ms": round(float(vals[min(int(n * 0.95), n - 1)]), 2),
            "p100_max_ms": round(float(max(vals)), 2),
            "min_ms": round(float(min(vals)), 2)
        }

    profile_payload = {
        "profile_metadata": {
            "model_e5": "intfloat/multilingual-e5-base",
            "model_crossencoder": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "candidate_k": 15,
            "rerank_top_k": 5,
            "device": retriever.device,
            "warm_samples_count": len(warm_measurements),
            "initialization_time_ms": round(init_time_ms, 1),
        },
        "cold_request": cold_profile,
        "warm_statistics": stats
    }

    PROFILE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(profile_payload, f, indent=2)

    with open(OPTIMIZED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(profile_payload, f, indent=2)

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'STAGE':<32} | {'MEAN (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'MAX (ms)'}")
    print("=" * 75)
    for k in keys:
        s = stats[k]
        print(f"{k:<32} | {s['mean_ms']:>8.2f} ms | {s['p50_ms']:>8.2f} ms | {s['p95_ms']:>8.2f} ms | {s['p100_max_ms']:>8.2f} ms")
    print("=" * 75)
    print(f"\nProfile saved to: {PROFILE_OUTPUT_FILE}\n")

if __name__ == "__main__":
    profile_pipeline()
