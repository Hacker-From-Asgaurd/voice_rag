import json
import numpy as np

EMBEDDINGS_FILE = "data/e5_embeddings.npy"
METADATA_FILE = "data/e5_metadata.json"


print("Loading E5 embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print("Embedding shape:", embeddings.shape)


print("\nLoading E5 metadata...")

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


print("\nChecking embedding dimension...")

if embeddings.shape[1] != 768:
    raise ValueError(
        f"Unexpected dimension: {embeddings.shape[1]}"
    )

print("Dimension check: PASSED")


print("\nChecking first record...")

print("Chunk:")
print(metadata[0]["chunk"][:300])

print("\nFirst 10 vector values:")
print(embeddings[0][:10])


print("\nAll E5 embedding checks passed.")