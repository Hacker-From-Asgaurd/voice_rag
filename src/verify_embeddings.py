import json
import numpy as np

EMBEDDINGS_FILE = "data/adaptive_embeddings.npy"
METADATA_FILE = "data/adaptive_metadata.json"

print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print("Embedding shape:", embeddings.shape)

print("\nLoading metadata...")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Metadata records:", len(metadata))

print("\nChecking alignment...")

if len(metadata) != embeddings.shape[0]:
    raise ValueError(
        f"Mismatch! Embeddings: {embeddings.shape[0]}, "
        f"Metadata: {len(metadata)}"
    )

print("Alignment check: PASSED")

print("\nChecking vector dimension...")

if embeddings.shape[1] != 384:
    raise ValueError(
        f"Unexpected embedding dimension: {embeddings.shape[1]}"
    )

print("Dimension check: PASSED")

print("\nChecking first record...")

print("Chunk:")
print(metadata[0]["chunk"][:300])

print("\nVector first 10 values:")
print(embeddings[0][:10])

print("\nAll embedding checks passed.")