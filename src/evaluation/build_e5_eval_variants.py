import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-base"

DATASET_FILE = "data/hindi_dev.parquet"

CORPORA = {
    "fixed": "data/chunks_fixed.jsonl",
    "sentence": "data/chunks_sentence.jsonl",
}

OUTPUT_DIR = Path("data/e5_eval_variants")

NUM_QUERIES = 50


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("BUILDING E5 EVALUATION VARIANTS")
print("=" * 70)


# ---------------------------------------------------------
# Load evaluation queries
# ---------------------------------------------------------

print("\nLoading evaluation dataset...")

df = pd.read_parquet(
    DATASET_FILE
).head(NUM_QUERIES)

evaluation_query_ids = {
    int(x)
    for x in df["query_id"].tolist()
}

print(
    f"Evaluation queries: {len(evaluation_query_ids)}"
)


# ---------------------------------------------------------
# Load E5
# ---------------------------------------------------------

print("\nLoading E5 model...")
    
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

print("E5 model loaded.")


# ---------------------------------------------------------
# Process fixed + sentence
# ---------------------------------------------------------

for strategy, input_file in CORPORA.items():

    print("\n" + "-" * 70)
    print(f"STRATEGY: {strategy.upper()}")
    print("-" * 70)

    print(
        f"Loading: {input_file}"
    )

    records = []

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            record = json.loads(line)

            if int(record["query_id"]) in evaluation_query_ids:
                records.append(record)

    print(
        f"Evaluation chunks: {len(records)}"
    )

    if not records:
        raise RuntimeError(
            f"No records found for {strategy}"
        )

    texts = [
        "passage: " + record["chunk"]
        for record in records
    ]

    print("Generating E5 embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    embedding_file = (
        OUTPUT_DIR /
        f"e5_{strategy}_embeddings.npy"
    )

    metadata_file = (
        OUTPUT_DIR /
        f"e5_{strategy}_metadata.json"
    )

    np.save(
        embedding_file,
        embeddings
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False
        )

    print(
        f"Shape: {embeddings.shape}"
    )

    print(
        f"Saved: {embedding_file}"
    )

    print(
        f"Saved: {metadata_file}"
    )


print("\n" + "=" * 70)
print("EVALUATION VARIANTS READY")
print("=" * 70)