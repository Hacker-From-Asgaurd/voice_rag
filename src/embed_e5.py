import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-base"

INPUT_FILE = "data/chunks_adaptive.jsonl"

OUTPUT_FILE = "data/e5_embeddings.npy"
METADATA_FILE = "data/e5_metadata.json"


print("Loading E5 embedding model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

print("Model loaded.")


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

print("\nLoading chunks...")

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print("Chunks loaded:", len(records))


# E5 retrieval convention:
# passages use "passage:" prefix

texts = [
    "passage: " + record["chunk"]
    for record in records
]


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

print("\nGenerating E5 embeddings...")

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# Convert to float32
# ---------------------------------------------------------

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


print("\nEmbedding shape:")
print(embeddings.shape)


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------

np.save(
    OUTPUT_FILE,
    embeddings
)


# ---------------------------------------------------------
# Save metadata
# ---------------------------------------------------------

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(
        records,
        f,
        ensure_ascii=False
    )


print("\nSaved:")
print(OUTPUT_FILE)
print(METADATA_FILE)