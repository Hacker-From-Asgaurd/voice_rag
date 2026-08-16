import json
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

INPUT_FILE = "data/chunks_adaptive.jsonl"
OUTPUT_FILE = "data/adaptive_embeddings.npy"
METADATA_FILE = "data/adaptive_metadata.json"


print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

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


texts = [record["chunk"] for record in records]


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# Convert to NumPy
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
