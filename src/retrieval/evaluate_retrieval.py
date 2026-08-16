import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-base"

INDEX_FILE = "data/e5_adaptive.index"
METADATA_FILE = "data/e5_metadata.json"
DATASET_FILE = "data/hindi_dev.parquet"

NUM_QUERIES = 50
TOP_K = 10


print("Loading E5 model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading metadata...")
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Loading evaluation dataset...")
df = pd.read_parquet(DATASET_FILE)

df = df.head(NUM_QUERIES)

print(f"Queries to evaluate: {len(df)}")


# ---------------------------------------------------------
# Build ground truth from the original dataset
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------

recall_at_1 = 0
recall_at_5 = 0
recall_at_10 = 0

mrr_total = 0.0


print("\nStarting evaluation...\n")


for query_number, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    query_id = int(row["query_id"])
    query = row["query"]

    relevant_passages = ground_truth[query_id]

    # -----------------------------------------------------
    # Create query embedding
    # -----------------------------------------------------

    query_embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Search the GLOBAL index.
    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    # -----------------------------------------------------
    # Determine relevance of each retrieved result
    # -----------------------------------------------------

    relevance = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0:
            continue

        record = metadata[idx]

        retrieved_query_id = int(
            record["query_id"]
        )

        retrieved_passage_id = int(
            record["passage_id"]
        )

        # A result is relevant only if it belongs
        # to this query AND its original passage
        # is marked selected.
        is_relevant = (
            retrieved_query_id == query_id
            and
            retrieved_passage_id in relevant_passages
        )

        relevance.append(
            is_relevant
        )

    # -----------------------------------------------------
    # Recall@1
    # -----------------------------------------------------

    if any(relevance[:1]):
        recall_at_1 += 1

    # -----------------------------------------------------
    # Recall@5
    # -----------------------------------------------------

    if any(relevance[:5]):
        recall_at_5 += 1

    # -----------------------------------------------------
    # Recall@10
    # -----------------------------------------------------

    if any(relevance[:10]):
        recall_at_10 += 1

    # -----------------------------------------------------
    # MRR@10
    # -----------------------------------------------------

    reciprocal_rank = 0.0

    for rank, is_relevant in enumerate(
        relevance,
        start=1
    ):

        if is_relevant:

            reciprocal_rank = 1.0 / rank
            break

    mrr_total += reciprocal_rank

    if query_number % 10 == 0:

        print(
            f"Processed "
            f"{query_number}/{len(df)} queries..."
        )


# ---------------------------------------------------------
# Calculate final metrics
# ---------------------------------------------------------

num_queries = len(df)

recall_at_1 /= num_queries
recall_at_5 /= num_queries
recall_at_10 /= num_queries

mrr = mrr_total / num_queries


print("\n" + "=" * 70)
print("CORRECTED RETRIEVAL EVALUATION")
print("=" * 70)

print(
    f"Queries evaluated : {num_queries}"
)

print(
    f"Recall@1          : {recall_at_1:.4f}"
)

print(
    f"Recall@5          : {recall_at_5:.4f}"
)

print(
    f"Recall@10         : {recall_at_10:.4f}"
)

print(
    f"MRR@10            : {mrr:.4f}"
)

print("=" * 70)