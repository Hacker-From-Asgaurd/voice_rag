import json
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "intfloat/multilingual-e5-base"

DATASET_FILE = "data/hindi_dev.parquet"

NUM_QUERIES = 50
TOP_K = 10

# Evaluation-subset embeddings
FIXED_EMBEDDINGS = "data/e5_eval_variants/e5_fixed_embeddings.npy"
FIXED_METADATA = "data/e5_eval_variants/e5_fixed_metadata.json"

SENTENCE_EMBEDDINGS = "data/e5_eval_variants/e5_sentence_embeddings.npy"
SENTENCE_METADATA = "data/e5_eval_variants/e5_sentence_metadata.json"

# Existing adaptive E5 baseline
ADAPTIVE_EMBEDDINGS = "data/e5_embeddings.npy"
ADAPTIVE_METADATA = "data/e5_metadata.json"


# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 70)
print("CHUNKING STRATEGY BENCHMARK")
print("=" * 70)

print("\nLoading evaluation dataset...")

df = pd.read_parquet(
    DATASET_FILE
).head(NUM_QUERIES)

print(
    f"Queries evaluated: {len(df)}"
)


# =========================================================
# GROUND TRUTH
# =========================================================

ground_truth = {}

for _, row in df.iterrows():

    query_id = int(row["query_id"])

    selected = row["passages"]["is_selected"]

    relevant_passages = {
        passage_id
        for passage_id, value in enumerate(selected)
        if int(value) == 1
    }

    ground_truth[query_id] = relevant_passages


# =========================================================
# LOAD E5 MODEL
# =========================================================

print("\nLoading E5 model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
)

print(
    "Device:",
    model.device
)


# =========================================================
# LOAD VARIANT
# =========================================================

def load_variant(
    name,
    embeddings_file,
    metadata_file
):

    print(
        f"\nLoading {name} variant..."
    )

    embeddings = np.load(
        embeddings_file
    ).astype("float32")

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    print(
        f"{name} vectors: {len(embeddings)}"
    )

    print(
        f"{name} metadata: {len(metadata)}"
    )

    if len(embeddings) != len(metadata):

        raise RuntimeError(
            f"{name}: embeddings and metadata "
            "length mismatch"
        )

    return embeddings, metadata


fixed_embeddings, fixed_metadata = load_variant(
    "FIXED",
    FIXED_EMBEDDINGS,
    FIXED_METADATA
)

sentence_embeddings, sentence_metadata = load_variant(
    "SENTENCE",
    SENTENCE_EMBEDDINGS,
    SENTENCE_METADATA
)

adaptive_embeddings, adaptive_metadata = load_variant(
    "ADAPTIVE",
    ADAPTIVE_EMBEDDINGS,
    ADAPTIVE_METADATA
)


# =========================================================
# BUILD FAISS INDEXES
# =========================================================

print("\nBuilding FAISS indexes...")

fixed_index = faiss.IndexFlatIP(
    fixed_embeddings.shape[1]
)

fixed_index.add(
    fixed_embeddings
)


sentence_index = faiss.IndexFlatIP(
    sentence_embeddings.shape[1]
)

sentence_index.add(
    sentence_embeddings
)


adaptive_index = faiss.IndexFlatIP(
    adaptive_embeddings.shape[1]
)

adaptive_index.add(
    adaptive_embeddings
)

print("FAISS indexes ready.")


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_strategy(
    strategy_name,
    index,
    metadata
):

    print("\n" + "-" * 70)
    print(
        f"EVALUATING: {strategy_name}"
    )
    print("-" * 70)

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    mrr_total = 0.0

    latencies = []

    for query_number, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        query_id = int(
            row["query_id"]
        )

        query = str(
            row["query"]
        )

        relevant_passages = (
            ground_truth[query_id]
        )

        # -------------------------------------------------
        # Query embedding
        # -------------------------------------------------

        start_time = time.perf_counter()

        query_embedding = model.encode(
            ["query: " + query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # -------------------------------------------------
        # FAISS search
        # -------------------------------------------------

        scores, indices = index.search(
            query_embedding,
            TOP_K
        )

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        latencies.append(
            elapsed_ms
        )

        # -------------------------------------------------
        # Relevance
        # -------------------------------------------------

        relevance = []

        for idx in indices[0]:

            if idx < 0:
                continue

            record = metadata[idx]

            retrieved_query_id = int(
                record["query_id"]
            )

            retrieved_passage_id = int(
                record["passage_id"]
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

        # -------------------------------------------------
        # Recall@1
        # -------------------------------------------------

        if any(relevance[:1]):
            recall_at_1 += 1

        # -------------------------------------------------
        # Recall@5
        # -------------------------------------------------

        if any(relevance[:5]):
            recall_at_5 += 1

        # -------------------------------------------------
        # Recall@10
        # -------------------------------------------------

        if any(relevance[:10]):
            recall_at_10 += 1

        # -------------------------------------------------
        # MRR@10
        # -------------------------------------------------

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

        if query_number % 10 == 0:

            print(
                f"Processed "
                f"{query_number}/{len(df)}"
            )

    num_queries = len(df)

    return {
        "recall_at_1":
            recall_at_1 / num_queries,

        "recall_at_5":
            recall_at_5 / num_queries,

        "recall_at_10":
            recall_at_10 / num_queries,

        "mrr":
            mrr_total / num_queries,

        "avg_latency":
            float(
                np.mean(latencies)
            ),

        "p50_latency":
            float(
                np.percentile(
                    latencies,
                    50
                )
            ),

        "p95_latency":
            float(
                np.percentile(
                    latencies,
                    95
                )
            )
    }


# =========================================================
# RUN EVALUATION
# =========================================================

results = {}

results["Fixed"] = evaluate_strategy(
    "FIXED",
    fixed_index,
    fixed_metadata
)

results["Sentence"] = evaluate_strategy(
    "SENTENCE-AWARE",
    sentence_index,
    sentence_metadata
)

results["Adaptive"] = evaluate_strategy(
    "ADAPTIVE PARENT-CHILD",
    adaptive_index,
    adaptive_metadata
)


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n")
print("=" * 70)
print("CHUNKING STRATEGY RESULTS")
print("=" * 70)

for strategy, result in results.items():

    print(
        f"\n{strategy.upper()}"
    )

    print(
        f"Recall@1  : "
        f"{result['recall_at_1']:.4f}"
    )

    print(
        f"Recall@5  : "
        f"{result['recall_at_5']:.4f}"
    )

    print(
        f"Recall@10 : "
        f"{result['recall_at_10']:.4f}"
    )

    print(
        f"MRR@10    : "
        f"{result['mrr']:.4f}"
    )

    print(
        f"Avg latency : "
        f"{result['avg_latency']:.2f} ms"
    )

    print(
        f"P50 latency : "
        f"{result['p50_latency']:.2f} ms"
    )

    print(
        f"P95 latency : "
        f"{result['p95_latency']:.2f} ms"
    )


# =========================================================
# BEST STRATEGY
# =========================================================

best_recall = max(
    results.items(),
    key=lambda item:
        item[1]["recall_at_10"]
)

best_mrr = max(
    results.items(),
    key=lambda item:
        item[1]["mrr"]
)


print("\n")
print("=" * 70)
print("BEST RESULTS")
print("=" * 70)

print(
    f"Best Recall@10 : "
    f"{best_recall[0]} "
    f"({best_recall[1]['recall_at_10']:.4f})"
)

print(
    f"Best MRR@10    : "
    f"{best_mrr[0]} "
    f"({best_mrr[1]['mrr']:.4f})"
)

print("=" * 70)

print(
    "\nNOTE:"
)

print(
    "Fixed and Sentence variants are evaluated "
    "using chunks belonging to the 50-query "
    "evaluation subset."
)

print(
    "Adaptive uses the existing full E5 corpus."
)

print(
    "Therefore, this is a controlled chunking "
    "ablation, not a full-corpus production benchmark."
)

print("=" * 70)