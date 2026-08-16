import json
import time
import statistics

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATASET_FILE = "data/hindi_dev.parquet"

ADAPTIVE_INDEX_FILE = "data/adaptive.index"
ADAPTIVE_METADATA_FILE = "data/adaptive_metadata.json"

E5_INDEX_FILE = "data/e5_adaptive.index"
E5_METADATA_FILE = "data/e5_metadata.json"

ADAPTIVE_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

E5_MODEL_NAME = (
    "intfloat/multilingual-e5-base"
)

NUM_QUERIES = 50
TOP_K = 10


def load_metadata(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def build_ground_truth(df):

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

    return ground_truth


def search(
    model,
    index,
    metadata,
    query,
    use_e5
):

    start = time.perf_counter()

    if use_e5:

        text = "query: " + query

    else:

        text = query

    embedding = model.encode(
        [text],
        normalize_embeddings=True
    )

    embedding = np.asarray(
        embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        embedding,
        TOP_K
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        record = metadata[idx]

        results.append({
            "score": float(score),
            "query_id": int(
                record["query_id"]
            ),
            "passage_id": int(
                record["passage_id"]
            ),
            "is_selected": int(
                record["is_selected"]
            )
        })

    return results, latency_ms


def evaluate(
    results_by_query,
    ground_truth
):

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    mrr_total = 0.0

    for query_id, results in (
        results_by_query.items()
    ):

        relevant_passages = (
            ground_truth[query_id]
        )

        relevance = []

        for result in results:

            is_relevant = (
                result["query_id"]
                == query_id
                and
                result["passage_id"]
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

        for rank, relevant in enumerate(
            relevance,
            start=1
        ):

            if relevant:

                reciprocal_rank = (
                    1.0 / rank
                )

                break

        mrr_total += reciprocal_rank

    total = len(
        results_by_query
    )

    return (
        recall_at_1 / total,
        recall_at_5 / total,
        recall_at_10 / total,
        mrr_total / total
    )


def main():

    print("=" * 70)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 70)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print("\nLoading evaluation dataset...")

    df = pd.read_parquet(
        DATASET_FILE
    )

    df = df.head(
        NUM_QUERIES
    )

    print(
        "Queries evaluated:",
        len(df)
    )

    ground_truth = (
        build_ground_truth(df)
    )

    # -----------------------------------------------------
    # Load adaptive model
    # -----------------------------------------------------

    print(
        "\nLoading adaptive "
        "embedding model..."
    )

    adaptive_model = (
        SentenceTransformer(
            ADAPTIVE_MODEL_NAME
        )
    )

    # -----------------------------------------------------
    # Load E5
    # -----------------------------------------------------

    print(
        "\nLoading E5 model..."
    )

    e5_model = (
        SentenceTransformer(
            E5_MODEL_NAME
        )
    )

    # -----------------------------------------------------
    # Load indexes
    # -----------------------------------------------------

    print(
        "\nLoading adaptive index..."
    )

    adaptive_index = (
        faiss.read_index(
            ADAPTIVE_INDEX_FILE
        )
    )

    adaptive_metadata = (
        load_metadata(
            ADAPTIVE_METADATA_FILE
        )
    )

    print(
        "Adaptive vectors:",
        adaptive_index.ntotal
    )

    print(
        "\nLoading E5 index..."
    )

    e5_index = (
        faiss.read_index(
            E5_INDEX_FILE
        )
    )

    e5_metadata = (
        load_metadata(
            E5_METADATA_FILE
        )
    )

    print(
        "E5 vectors:",
        e5_index.ntotal
    )

    # -----------------------------------------------------
    # Benchmark
    # -----------------------------------------------------

    adaptive_results = {}
    e5_results = {}

    adaptive_latencies = []
    e5_latencies = []

    print(
        "\nStarting comparison...\n"
    )

    for number, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        query_id = int(
            row["query_id"]
        )

        query = row["query"]

        print(
            f"[{number}/{len(df)}] "
            f"{query}"
        )

        # -----------------------------------------------
        # Adaptive MiniLM
        # -----------------------------------------------

        results, latency = search(
            adaptive_model,
            adaptive_index,
            adaptive_metadata,
            query,
            use_e5=False
        )

        adaptive_results[
            query_id
        ] = results

        adaptive_latencies.append(
            latency
        )

        # -----------------------------------------------
        # E5
        # -----------------------------------------------

        results, latency = search(
            e5_model,
            e5_index,
            e5_metadata,
            query,
            use_e5=True
        )

        e5_results[
            query_id
        ] = results

        e5_latencies.append(
            latency
        )

        if number % 10 == 0:

            print(
                f"Processed "
                f"{number}/{len(df)} queries..."
            )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    adaptive_metrics = evaluate(
        adaptive_results,
        ground_truth
    )

    e5_metrics = evaluate(
        e5_results,
        ground_truth
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PARAPHRASE-MULTILINGUAL-MINILM")
    print("=" * 70)

    print(
        f"Recall@1  : "
        f"{adaptive_metrics[0]:.4f}"
    )

    print(
        f"Recall@5  : "
        f"{adaptive_metrics[1]:.4f}"
    )

    print(
        f"Recall@10 : "
        f"{adaptive_metrics[2]:.4f}"
    )

    print(
        f"MRR@10    : "
        f"{adaptive_metrics[3]:.4f}"
    )

    print(
        f"Avg latency : "
        f"{statistics.mean(adaptive_latencies):.2f} ms"
    )

    print("\n" + "=" * 70)
    print("MULTILINGUAL E5")
    print("=" * 70)

    print(
        f"Recall@1  : "
        f"{e5_metrics[0]:.4f}"
    )

    print(
        f"Recall@5  : "
        f"{e5_metrics[1]:.4f}"
    )

    print(
        f"Recall@10 : "
        f"{e5_metrics[2]:.4f}"
    )

    print(
        f"MRR@10    : "
        f"{e5_metrics[3]:.4f}"
    )

    print(
        f"Avg latency : "
        f"{statistics.mean(e5_latencies):.2f} ms"
    )

    print("\n" + "=" * 70)
    print("E5 vs MINILM")
    print("=" * 70)

    print(
        f"Recall@1  : "
        f"{adaptive_metrics[0]:.4f} → "
        f"{e5_metrics[0]:.4f} "
        f"({e5_metrics[0] - adaptive_metrics[0]:+.4f})"
    )

    print(
        f"Recall@5  : "
        f"{adaptive_metrics[1]:.4f} → "
        f"{e5_metrics[1]:.4f} "
        f"({e5_metrics[1] - adaptive_metrics[1]:+.4f})"
    )

    print(
        f"Recall@10 : "
        f"{adaptive_metrics[2]:.4f} → "
        f"{e5_metrics[2]:.4f} "
        f"({e5_metrics[2] - adaptive_metrics[2]:+.4f})"
    )

    print(
        f"MRR@10    : "
        f"{adaptive_metrics[3]:.4f} → "
        f"{e5_metrics[3]:.4f} "
        f"({e5_metrics[3] - adaptive_metrics[3]:+.4f})"
    )

    print(
        f"Latency   : "
        f"{statistics.mean(adaptive_latencies):.2f} ms → "
        f"{statistics.mean(e5_latencies):.2f} ms"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()