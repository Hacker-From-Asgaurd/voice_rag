import json
import time

import faiss
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

INDEX_FILE = "data/e5_adaptive.index"
METADATA_FILE = "data/e5_metadata.json"
DATASET_FILE = "data/hindi_dev.parquet"

NUM_QUERIES = 50
FINAL_K = 10

CANDIDATE_KS = [5, 10, 20]


# =========================================================
# DEVICE
# =========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("RERANKER CANDIDATE-SIZE BENCHMARK")
print("=" * 70)

print("\nDevice:", DEVICE)


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading evaluation dataset...")

df = pd.read_parquet(
    DATASET_FILE
).head(NUM_QUERIES)

print(
    "Queries to evaluate:",
    len(df)
)


# =========================================================
# GROUND TRUTH
# =========================================================

ground_truth = {}

for _, row in df.iterrows():

    query_id = int(
        row["query_id"]
    )

    selected = row["passages"]["is_selected"]

    relevant = {
        passage_id
        for passage_id, value in enumerate(selected)
        if int(value) == 1
    }

    ground_truth[query_id] = relevant


# =========================================================
# LOAD E5
# =========================================================

print("\nLoading E5 model...")

embedder = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

print("E5 ready.")


# =========================================================
# LOAD FAISS
# =========================================================

print("\nLoading FAISS index...")

index = faiss.read_index(
    INDEX_FILE
)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print(
    "Vectors:",
    index.ntotal
)


# =========================================================
# LOAD RERANKER
# =========================================================

print("\nLoading reranker...")

reranker = CrossEncoder(
    RERANKER_NAME,
    device=DEVICE
)

print("Reranker ready.")


# =========================================================
# METRICS
# =========================================================

def evaluate_results(
    results,
    relevant_passages
):

    relevance = []

    for result in results:

        is_relevant = (
            result["passage_id"]
            in relevant_passages
        )

        relevance.append(
            is_relevant
        )

    recall_1 = (
        1
        if any(relevance[:1])
        else 0
    )

    recall_5 = (
        1
        if any(relevance[:5])
        else 0
    )

    recall_10 = (
        1
        if any(relevance[:10])
        else 0
    )

    reciprocal_rank = 0.0

    for rank, relevant in enumerate(
        relevance[:10],
        start=1
    ):

        if relevant:

            reciprocal_rank = (
                1.0 / rank
            )

            break

    return (
        recall_1,
        recall_5,
        recall_10,
        reciprocal_rank
    )


# =========================================================
# BASELINE
# =========================================================

print("\n" + "-" * 70)
print("FAISS BASELINE")
print("-" * 70)

baseline = {
    "r1": 0,
    "r5": 0,
    "r10": 0,
    "mrr": 0.0
}

for query_number, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    query_id = int(
        row["query_id"]
    )

    query = row["query"]

    relevant = ground_truth[
        query_id
    ]

    embedding = embedder.encode(
        ["query: " + query],
        normalize_embeddings=True
    )

    embedding = np.asarray(
        embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        embedding,
        FINAL_K
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        record = metadata[idx]

        results.append({
            "passage_id": int(
                record["passage_id"]
            )
        })

    r1, r5, r10, mrr = evaluate_results(
        results,
        relevant
    )

    baseline["r1"] += r1
    baseline["r5"] += r5
    baseline["r10"] += r10
    baseline["mrr"] += mrr


num_queries = len(df)

for key in baseline:

    baseline[key] /= num_queries


# =========================================================
# RERANKER EXPERIMENT
# =========================================================

all_results = {}


for candidate_k in CANDIDATE_KS:

    print("\n" + "-" * 70)
    print(
        f"RERANKER WITH TOP-{candidate_k} CANDIDATES"
    )
    print("-" * 70)

    metrics = {
        "r1": 0,
        "r5": 0,
        "r10": 0,
        "mrr": 0.0,
        "latencies": []
    }

    for query_number, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        query_id = int(
            row["query_id"]
        )

        query = row["query"]

        relevant = ground_truth[
            query_id
        ]

        # -------------------------------------------------
        # E5 retrieval
        # -------------------------------------------------

        embedding = embedder.encode(
            ["query: " + query],
            normalize_embeddings=True
        )

        embedding = np.asarray(
            embedding,
            dtype="float32"
        )

        scores, indices = index.search(
            embedding,
            candidate_k
        )

        candidates = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            record = metadata[idx]

            candidates.append({
                "passage_id": int(
                    record["passage_id"]
                ),
                "chunk": record["chunk"],
                "faiss_score": float(score)
            })

        # -------------------------------------------------
        # Rerank
        # -------------------------------------------------

        pairs = [
            [query, candidate["chunk"]]
            for candidate in candidates
        ]

        start = time.perf_counter()

        rerank_scores = reranker.predict(
            pairs,
            show_progress_bar=False
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        metrics["latencies"].append(
            elapsed
        )

        ranked = sorted(
            zip(
                candidates,
                rerank_scores
            ),
            key=lambda x: float(x[1]),
            reverse=True
        )

        results = []

        for candidate, score in ranked:

            results.append({
                "passage_id":
                    candidate["passage_id"]
            })

        results = results[:FINAL_K]

        r1, r5, r10, mrr = evaluate_results(
            results,
            relevant
        )

        metrics["r1"] += r1
        metrics["r5"] += r5
        metrics["r10"] += r10
        metrics["mrr"] += mrr

        if query_number % 10 == 0:

            print(
                f"Processed "
                f"{query_number}/{num_queries}"
            )

    metrics["r1"] /= num_queries
    metrics["r5"] /= num_queries
    metrics["r10"] /= num_queries
    metrics["mrr"] /= num_queries

    latencies = np.asarray(
        metrics["latencies"]
    )

    metrics["avg"] = float(
        np.mean(latencies)
    )

    metrics["p50"] = float(
        np.percentile(
            latencies,
            50
        )
    )

    metrics["p95"] = float(
        np.percentile(
            latencies,
            95
        )
    )

    metrics["p100"] = float(
        np.max(latencies)
    )

    all_results[candidate_k] = metrics


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 70)
print("RERANKER CANDIDATE-SIZE RESULTS")
print("=" * 70)

print("\nFAISS BASELINE")

print(
    f"Recall@1  : {baseline['r1']:.4f}"
)

print(
    f"Recall@5  : {baseline['r5']:.4f}"
)

print(
    f"Recall@10 : {baseline['r10']:.4f}"
)

print(
    f"MRR@10    : {baseline['mrr']:.4f}"
)


for candidate_k, metrics in all_results.items():

    print("\n" + "-" * 70)

    print(
        f"TOP-{candidate_k} → RERANK → TOP-{FINAL_K}"
    )

    print("-" * 70)

    print(
        f"Recall@1  : {metrics['r1']:.4f}"
    )

    print(
        f"Recall@5  : {metrics['r5']:.4f}"
    )

    print(
        f"Recall@10 : {metrics['r10']:.4f}"
    )

    print(
        f"MRR@10    : {metrics['mrr']:.4f}"
    )

    print(
        f"Avg       : {metrics['avg']:.2f} ms"
    )

    print(
        f"P50       : {metrics['p50']:.2f} ms"
    )

    print(
        f"P95       : {metrics['p95']:.2f} ms"
    )

    print(
        f"P100      : {metrics['p100']:.2f} ms"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)