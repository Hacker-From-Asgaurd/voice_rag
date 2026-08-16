import sys
import json
import time
import statistics
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


DATASET_FILE = "data/hindi_dev.parquet"

NUM_QUERIES = 50

# Retrieve more candidates before reranking.
INITIAL_TOP_K = 10

# Number passed forward after reranking.
FINAL_TOP_K = 10


def evaluate_results(
    results,
    ground_truth
):

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    mrr_total = 0.0

    num_queries = len(results)

    for query_id, retrieved in results.items():

        relevant_passages = ground_truth[
            query_id
        ]

        relevance = []

        for result in retrieved:

            retrieved_query_id = int(
                result["query_id"]
            )

            retrieved_passage_id = int(
                result["passage_id"]
            )

            is_relevant = (
                retrieved_query_id == query_id
                and
                retrieved_passage_id
                in relevant_passages
            )

            relevance.append(
                is_relevant
            )

        if any(relevance[:1]):
            recall_at_1 += 1

        if any(relevance[:5]):
            recall_at_5 += 1

        if any(relevance[:10]):
            recall_at_10 += 1

        reciprocal_rank = 0.0

        for rank, is_relevant in enumerate(
            relevance,
            start=1
        ):

            if is_relevant:

                reciprocal_rank = (
                    1.0 / rank
                )

                break

        mrr_total += reciprocal_rank

    if num_queries == 0:

        return 0, 0, 0, 0

    return (
        recall_at_1 / num_queries,
        recall_at_5 / num_queries,
        recall_at_10 / num_queries,
        mrr_total / num_queries
    )


def percentile(
    values,
    percentage
):

    if not values:
        return 0.0

    values = sorted(values)

    index = (
        (len(values) - 1)
        * percentage
        / 100
    )

    lower = int(index)

    upper = min(
        lower + 1,
        len(values) - 1
    )

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower]
        + weight
        * (
            values[upper]
            - values[lower]
        )
    )


def main():

    print("=" * 70)
    print("RERANKER BENCHMARK")
    print("=" * 70)

    print("\nLoading evaluation dataset...")

    df = pd.read_parquet(
        DATASET_FILE
    )

    df = df.head(
        NUM_QUERIES
    )

    print(
        f"Queries to evaluate: "
        f"{len(df)}"
    )

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    ground_truth = {}

    for _, row in df.iterrows():

        query_id = int(
            row["query_id"]
        )

        selected = (
            row["passages"]["is_selected"]
        )

        relevant_passages = {
            passage_id
            for passage_id, value
            in enumerate(selected)
            if int(value) == 1
        }

        ground_truth[
            query_id
        ] = relevant_passages

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    print("\nLoading retriever...")

    retriever = Retriever()

    print("\nLoading reranker...")

    reranker = Reranker()

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    baseline_results = {}
    reranked_results = {}

    rerank_latencies = []

    print("\nStarting benchmark...\n")

    for query_number, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        query_id = int(
            row["query_id"]
        )

        query = row["query"]

        print(
            f"[{query_number}/{len(df)}] "
            f"{query}"
        )

        # -------------------------------------------------
        # FAISS
        # -------------------------------------------------

        retrieved = retriever.search(
            query,
            top_k=INITIAL_TOP_K
        )

        baseline_results[
            query_id
        ] = retrieved[:10]

        # -------------------------------------------------
        # RERANK
        # -------------------------------------------------

        start_time = time.perf_counter()

        reranked = reranker.rerank(
            query,
            retrieved,
            top_k=FINAL_TOP_K
        )

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        rerank_latencies.append(
            elapsed_ms
        )

        reranked_results[
            query_id
        ] = reranked

        print(
            f"Candidates : {len(retrieved)}"
        )

        print(
            f"Rerank     : "
            f"{elapsed_ms:.2f} ms"
        )

        if query_number % 10 == 0:

            print(
                f"Processed "
                f"{query_number}/"
                f"{len(df)} queries..."
            )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    (
        baseline_r1,
        baseline_r5,
        baseline_r10,
        baseline_mrr
    ) = evaluate_results(
        baseline_results,
        ground_truth
    )

    (
        rerank_r1,
        rerank_r5,
        rerank_r10,
        rerank_mrr
    ) = evaluate_results(
        reranked_results,
        ground_truth
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FAISS BASELINE")
    print("=" * 70)

    print(
        f"Recall@1  : "
        f"{baseline_r1:.4f}"
    )

    print(
        f"Recall@5  : "
        f"{baseline_r5:.4f}"
    )

    print(
        f"Recall@10 : "
        f"{baseline_r10:.4f}"
    )

    print(
        f"MRR@10    : "
        f"{baseline_mrr:.4f}"
    )

    print("\n" + "=" * 70)
    print("FAISS + RERANKER")
    print("=" * 70)

    print(
        f"Recall@1  : "
        f"{rerank_r1:.4f}"
    )

    print(
        f"Recall@5  : "
        f"{rerank_r5:.4f}"
    )

    print(
        f"Recall@10 : "
        f"{rerank_r10:.4f}"
    )

    print(
        f"MRR@10    : "
        f"{rerank_mrr:.4f}"
    )

    print("\n" + "=" * 70)
    print("RERANKER LATENCY")
    print("=" * 70)

    print(
        f"Average : "
        f"{statistics.mean(rerank_latencies):.2f} ms"
    )

    print(
        f"P50     : "
        f"{percentile(rerank_latencies, 50):.2f} ms"
    )

    print(
        f"P95     : "
        f"{percentile(rerank_latencies, 95):.2f} ms"
    )

    print(
        f"P100    : "
        f"{max(rerank_latencies):.2f} ms"
    )

    print("\n" + "=" * 70)
    print("IMPROVEMENT")
    print("=" * 70)

    print(
        f"Recall@1  : "
        f"{baseline_r1:.4f} → "
        f"{rerank_r1:.4f} "
        f"({rerank_r1 - baseline_r1:+.4f})"
    )

    print(
        f"Recall@5  : "
        f"{baseline_r5:.4f} → "
        f"{rerank_r5:.4f} "
        f"({rerank_r5 - baseline_r5:+.4f})"
    )

    print(
        f"Recall@10 : "
        f"{baseline_r10:.4f} → "
        f"{rerank_r10:.4f} "
        f"({rerank_r10 - baseline_r10:+.4f})"
    )

    print(
        f"MRR@10    : "
        f"{baseline_mrr:.4f} → "
        f"{rerank_mrr:.4f} "
        f"({rerank_mrr - baseline_mrr:+.4f})"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()