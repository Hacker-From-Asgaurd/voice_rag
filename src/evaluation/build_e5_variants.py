import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-base"

CORPORA = {
    "fixed": "data/chunks_fixed.jsonl",
    "sentence": "data/chunks_sentence.jsonl",
    "adaptive": "data/chunks_adaptive.jsonl",
}

OUTPUT_DIR = Path("data/e5_variants")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("BUILDING E5 EMBEDDINGS FOR CHUNKING VARIANTS")
print("=" * 70)

print("\nLoading E5 model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("E5 model loaded.")


for strategy, input_file in CORPORA.items():

    print("\n" + "-" * 70)
    print(f"STRATEGY: {strategy.upper()}")
    print("-" * 70)

    print(
        f"Loading corpus: {input_file}"
    )

    records = []

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            records.append(
                json.loads(line)
            )

    print(
        f"Chunks loaded: {len(records)}"
    )

    texts = [
        "passage: " + record["chunk"]
        for record in records
    ]

    print("Generating E5 embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
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
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Saved embeddings: {embedding_file}"
    )

    print(
        f"Saved metadata:   {metadata_file}"
    )


print("\n" + "=" * 70)
print("ALL E5 VARIANTS BUILT")
print("=" * 70)