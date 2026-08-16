import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker

DATASET_FILE = Path("data/hindi_dev.parquet")
OUTPUT_STATS_FILE = Path("data/retrieval_benchmark_3037.json")
RERANK_CANDIDATES = 10


def compute_metrics(latencies, recall_1, recall_3, recall_5, mrr_total, total):
    latencies_sorted = sorted(latencies)
    avg_lat = sum(latencies_sorted) / total if total > 0 else 0.0
    p50_idx = min(int(total * 0.50), total - 1) if total > 0 else 0
    p70_idx = min(int(total * 0.70), total - 1) if total > 0 else 0
    p95_idx = min(int(total * 0.95), total - 1) if total > 0 else 0

    p50_lat = latencies_sorted[p50_idx] if total > 0 else 0.0
    p70_lat = latencies_sorted[p70_idx] if total > 0 else 0.0
    p95_lat = latencies_sorted[p95_idx] if total > 0 else 0.0
    p100_lat = max(latencies_sorted) if total > 0 else 0.0

    return {
        "queries": total,
        "recall_1": (recall_1 / total * 100) if total > 0 else 0.0,
        "recall_3": (recall_3 / total * 100) if total > 0 else 0.0,
        "recall_5": (recall_5 / total * 100) if total > 0 else 0.0,
        "mrr": (mrr_total / total) if total > 0 else 0.0,
        "avg_latency_ms": avg_lat,
        "p50_latency_ms": p50_lat,
        "p70_latency_ms": p70_lat,
        "p95_latency_ms": p95_lat,
        "p100_latency_ms": p100_lat,
    }


def evaluate_retrieval(df_valid, retriever, reranker=None):
    total = len(df_valid)
    recall_1 = 0
    recall_3 = 0
    recall_5 = 0
    mrr_total = 0.0
    latencies = []
    per_query_records = []

    for i, (_, row) in enumerate(df_valid.iterrows(), start=1):
        query_id = int(row["query_id"])
        query = row["query"]
        passages = row["passages"]

        selected_passage_ids = {
            pid
            for pid, flag in enumerate(passages["is_selected"])
            if int(flag) == 1
        }

        start_time = time.perf_counter()

        if reranker is None:
            # SYSTEM A: E5 Only (Top 5)
            results = retriever.search(query, top_k=5)
        else:
            # SYSTEM B: E5 Retrieval (Top 10) + CrossEncoder Reranking (Top 5)
            candidates = retriever.search(query, top_k=RERANK_CANDIDATES)
            results = reranker.rerank(query, candidates, top_k=5)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed_ms)

        # Ground truth verification:
        # Match both query_id and passage_id against ground truth selected passages
        relevance = []
        retrieved_ids = []
        for rank, res in enumerate(results, start=1):
            ret_qid = int(res["query_id"])
            ret_pid = int(res["passage_id"])
            is_rel = (ret_qid == query_id and ret_pid in selected_passage_ids)
            relevance.append(is_rel)
            retrieved_ids.append({"query_id": ret_qid, "passage_id": ret_pid, "is_rel": is_rel})

        r1 = any(relevance[:1])
        r3 = any(relevance[:3])
        r5 = any(relevance[:5])

        if r1:
            recall_1 += 1
        if r3:
            recall_3 += 1
        if r5:
            recall_5 += 1

        rr = 0.0
        for rank, is_rel in enumerate(relevance, start=1):
            if is_rel:
                rr = 1.0 / rank
                break
        mrr_total += rr

        per_query_records.append({
            "query_id": query_id,
            "latency_ms": elapsed_ms,
            "hit_at_1": r1,
            "hit_at_3": r3,
            "hit_at_5": r5,
            "reciprocal_rank": rr,
        })

    metrics = compute_metrics(latencies, recall_1, recall_3, recall_5, mrr_total, total)
    return metrics, per_query_records


def main():
    if not DATASET_FILE.exists():
        print(f"Error: Dataset not found at {DATASET_FILE}")
        return

    df = pd.read_parquet(
        DATASET_FILE,
        columns=["query_id", "query", "Answer", "passages"]
    )

    answerable = df[
        ~df["Answer"].astype(str).str.contains(
            "No Answer Present|कोई उत्तर नहीं मिला",
            case=False,
            regex=True
        )
    ]

    valid = answerable[
        answerable["passages"].apply(
            lambda x: 1 in list(x["is_selected"])
        )
    ]

    total_rows = len(df)
    answerable_count = len(answerable)
    valid_count = len(valid)

    # Initialize models (loading time NOT counted in per-query latency)
    retriever = Retriever()
    # Warm-up call
    retriever.search("परीक्षण", top_k=5)

    e5_metrics, e5_per_query = evaluate_retrieval(valid, retriever, reranker=None)

    reranker = Reranker()
    # Warm-up call
    candidates = retriever.search("परीक्षण", top_k=RERANK_CANDIDATES)
    reranker.rerank("परीक्षण", candidates, top_k=5)

    rerank_metrics, rerank_per_query = evaluate_retrieval(valid, retriever, reranker=reranker)

    # Print clean exact output
    print("=" * 60)
    print("FULL RETRIEVAL BENCHMARK")
    print("=" * 60)
    print(f"Dataset rows       : {total_rows}")
    print(f"Answerable         : {answerable_count}")
    print(f"Valid benchmark    : {valid_count}")
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20}{'E5 Only':<15}{'E5 + Reranker':<15}")
    print("-" * 60)
    print(f"{'Queries':<20}{e5_metrics['queries']:<15}{rerank_metrics['queries']:<15}")
    print(f"{'Recall@1':<20}{e5_metrics['recall_1']:.2f}%{'':<9}{rerank_metrics['recall_1']:.2f}%")
    print(f"{'Recall@3':<20}{e5_metrics['recall_3']:.2f}%{'':<9}{rerank_metrics['recall_3']:.2f}%")
    print(f"{'Recall@5':<20}{e5_metrics['recall_5']:.2f}%{'':<9}{rerank_metrics['recall_5']:.2f}%")
    print(f"{'MRR':<20}{e5_metrics['mrr']:.4f}{'':<9}{rerank_metrics['mrr']:.4f}")
    print(f"{'Avg latency':<20}{e5_metrics['avg_latency_ms']:.2f} ms{'':<6}{rerank_metrics['avg_latency_ms']:.2f} ms")
    print(f"{'P50 latency':<20}{e5_metrics['p50_latency_ms']:.2f} ms{'':<6}{rerank_metrics['p50_latency_ms']:.2f} ms")
    print(f"{'P70 latency':<20}{e5_metrics['p70_latency_ms']:.2f} ms{'':<6}{rerank_metrics['p70_latency_ms']:.2f} ms")
    print(f"{'P95 latency':<20}{e5_metrics['p95_latency_ms']:.2f} ms{'':<6}{rerank_metrics['p95_latency_ms']:.2f} ms")
    print(f"{'P100 latency':<20}{e5_metrics['p100_latency_ms']:.2f} ms{'':<6}{rerank_metrics['p100_latency_ms']:.2f} ms")
    print("-" * 60)

    output_data = {
        "dataset_counts": {
            "total_rows": total_rows,
            "answerable_rows": answerable_count,
            "valid_benchmark_queries": valid_count,
        },
        "benchmark_configuration": {
            "embedding_model": "intfloat/multilingual-e5-base",
            "index_type": "IndexFlatIP",
            "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "rerank_candidates": RERANK_CANDIDATES,
            "top_k_final": 5,
        },
        "metrics_e5_only": e5_metrics,
        "metrics_e5_plus_reranker": rerank_metrics,
        "per_query_e5_only": e5_per_query,
        "per_query_e5_plus_reranker": rerank_per_query,
    }

    OUTPUT_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_STATS_FILE}")


if __name__ == "__main__":
    main()
