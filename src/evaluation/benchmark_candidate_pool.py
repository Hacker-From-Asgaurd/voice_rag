import json
import sys
import time
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker

DATASET_FILE = Path("data/hindi_dev.parquet")
OUTPUT_STATS_FILE = Path("data/candidate_pool_benchmark.json")
CANDIDATE_SIZES = [10, 15, 20, 25, 30]


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


def main():
    if not DATASET_FILE.exists():
        print(f"Error: Dataset not found at {DATASET_FILE}")
        return

    print("Loading dataset...")
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

    total = len(valid)
    print(f"Valid benchmark queries: {total}")

    print("Loading Retriever and Reranker on CUDA...")
    retriever = Retriever()
    reranker = Reranker()

    # Warmup
    warmup_cands = retriever.search("परीक्षण", top_k=30)
    reranker.rerank("परीक्षण", warmup_cands[:10], top_k=5)

    all_k_metrics = {}
    latencies_by_k = {k: [] for k in CANDIDATE_SIZES}
    recall_1_by_k = {k: 0 for k in CANDIDATE_SIZES}
    recall_3_by_k = {k: 0 for k in CANDIDATE_SIZES}
    recall_5_by_k = {k: 0 for k in CANDIDATE_SIZES}
    mrr_by_k = {k: 0.0 for k in CANDIDATE_SIZES}

    print("Running single-pass candidate depth evaluation across all 3,037 queries...")

    for i, (_, row) in enumerate(valid.iterrows(), start=1):
        query_id = int(row["query_id"])
        query = row["query"]
        passages = row["passages"]

        selected_passage_ids = {
            pid
            for pid, flag in enumerate(passages["is_selected"])
            if int(flag) == 1
        }

        # Step 1: E5 search top-30
        e5_start = time.perf_counter()
        candidates_30 = retriever.search(query, top_k=30)
        e5_elapsed = (time.perf_counter() - e5_start) * 1000

        # Step 2: Evaluate each candidate pool size k
        for k in CANDIDATE_SIZES:
            k_candidates = candidates_30[:k]
            rerank_start = time.perf_counter()
            results = reranker.rerank(query, k_candidates, top_k=5)
            rerank_elapsed = (time.perf_counter() - rerank_start) * 1000

            total_elapsed = e5_elapsed + rerank_elapsed
            latencies_by_k[k].append(total_elapsed)

            relevance = []
            for res in results:
                ret_qid = int(res["query_id"])
                ret_pid = int(res["passage_id"])
                is_rel = (ret_qid == query_id and ret_pid in selected_passage_ids)
                relevance.append(is_rel)

            if any(relevance[:1]):
                recall_1_by_k[k] += 1
            if any(relevance[:3]):
                recall_3_by_k[k] += 1
            if any(relevance[:5]):
                recall_5_by_k[k] += 1

            rr = 0.0
            for rank, is_rel in enumerate(relevance, start=1):
                if is_rel:
                    rr = 1.0 / rank
                    break
            mrr_by_k[k] += rr

        if i % 600 == 0 or i == total:
            print(f"  Processed {i}/{total} queries ({i/total*100:.1f}%)...")

    for k in CANDIDATE_SIZES:
        metrics = compute_metrics(
            latencies_by_k[k],
            recall_1_by_k[k],
            recall_3_by_k[k],
            recall_5_by_k[k],
            mrr_by_k[k],
            total,
        )
        all_k_metrics[f"k_{k}"] = metrics

    # Output clean summary table
    print("\n" + "=" * 80)
    print("PHASE 3: CANDIDATE POOL SCALING EXPERIMENT (ALL 3,037 QUERIES)")
    print("=" * 80)
    header = f"{'Metric':<18}" + "".join([f"{'k=' + str(k):<12}" for k in CANDIDATE_SIZES])
    print(header)
    print("-" * 80)

    for name, key, is_pct, is_float in [
        ("Recall@1", "recall_1", True, False),
        ("Recall@3", "recall_3", True, False),
        ("Recall@5", "recall_5", True, False),
        ("MRR", "mrr", False, True),
        ("Avg Latency", "avg_latency_ms", False, False),
        ("P50 Latency", "p50_latency_ms", False, False),
        ("P70 Latency", "p70_latency_ms", False, False),
        ("P95 Latency", "p95_latency_ms", False, False),
        ("P100 Latency", "p100_latency_ms", False, False),
    ]:
        row_str = f"{name:<18}"
        for k in CANDIDATE_SIZES:
            val = all_k_metrics[f"k_{k}"][key]
            if is_pct:
                row_str += f"{val:.2f}%{'':<6}"
            elif is_float:
                row_str += f"{val:.4f}{'':<6}"
            else:
                row_str += f"{val:.1f} ms{'':<5}"
        print(row_str)

    print("=" * 80)

    OUTPUT_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_k_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_STATS_FILE}")


if __name__ == "__main__":
    main()
