import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-base"

DATASET_FILE = "data/hindi_dev.parquet"

NUM_QUERIES = 50


print("Loading E5 model...")

model = SentenceTransformer(MODEL_NAME)

print("Loading dataset...")

df = pd.read_parquet(DATASET_FILE).head(NUM_QUERIES)

print(f"Queries to evaluate: {len(df)}")


recall_at_1 = 0
recall_at_5 = 0
recall_at_10 = 0

mrr_total = 0.0


print("\nStarting diagnostic evaluation...\n")


for query_number, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    query = row["query"]

    passages = row["passages"]

    translated_passages = passages["Translated_passages"]

    selected = passages["is_selected"]


    # -----------------------------------------------------
    # Remove empty passages
    # -----------------------------------------------------

    candidates = []

    for passage_id, passage in enumerate(
        translated_passages
    ):

        if passage is None:
            continue

        passage = str(passage).strip()

        if not passage:
            continue

        candidates.append(
            {
                "passage_id": passage_id,
                "text": passage,
                "selected": int(selected[passage_id])
            }
        )


    if not candidates:
        continue


    # -----------------------------------------------------
    # Encode query
    # -----------------------------------------------------

    query_embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0]


    # -----------------------------------------------------
    # Encode candidate passages
    # -----------------------------------------------------

    passage_texts = [
        "passage: " + item["text"]
        for item in candidates
    ]

    passage_embeddings = model.encode(
        passage_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )


    # -----------------------------------------------------
    # Cosine similarity
    # -----------------------------------------------------

    scores = np.dot(
        passage_embeddings,
        query_embedding
    )


    # -----------------------------------------------------
    # Sort by similarity
    # -----------------------------------------------------

    ranked_indices = np.argsort(
        scores
    )[::-1]


    ranked_candidates = [
        candidates[i]
        for i in ranked_indices
    ]


    # -----------------------------------------------------
    # Find first relevant passage
    # -----------------------------------------------------

    relevant_ranks = []

    for rank, candidate in enumerate(
        ranked_candidates,
        start=1
    ):

        if candidate["selected"] == 1:

            relevant_ranks.append(rank)


    if not relevant_ranks:
        continue


    first_relevant_rank = relevant_ranks[0]


    # -----------------------------------------------------
    # Recall@K
    # -----------------------------------------------------

    if first_relevant_rank <= 1:
        recall_at_1 += 1

    if first_relevant_rank <= 5:
        recall_at_5 += 1

    if first_relevant_rank <= 10:
        recall_at_10 += 1


    # -----------------------------------------------------
    # MRR
    # -----------------------------------------------------

    mrr_total += 1.0 / first_relevant_rank


    if query_number % 10 == 0:

        print(
            f"Processed "
            f"{query_number}/{len(df)} queries..."
        )


# ---------------------------------------------------------
# Final results
# ---------------------------------------------------------

num_queries = len(df)

recall_at_1 /= num_queries
recall_at_5 /= num_queries
recall_at_10 /= num_queries

mrr = mrr_total / num_queries


print("\n" + "=" * 70)
print("DIAGNOSTIC RETRIEVAL EVALUATION")
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
    f"MRR               : {mrr:.4f}"
)

print("=" * 70)