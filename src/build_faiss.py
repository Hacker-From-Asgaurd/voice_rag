import faiss
import numpy as np

EMBEDDINGS_FILE = "data/adaptive_embeddings.npy"
INDEX_FILE = "data/adaptive.index"


print("Loading embeddings...")

embeddings = np.load(
    EMBEDDINGS_FILE
).astype("float32")

print("Embedding shape:", embeddings.shape)


# ---------------------------------------------------------
# Create FAISS index
# ---------------------------------------------------------

dimension = embeddings.shape[1]

print("\nCreating FAISS index...")
print("Vector dimension:", dimension)

index = faiss.IndexFlatIP(dimension)


# ---------------------------------------------------------
# Add vectors
# ---------------------------------------------------------

print("\nAdding vectors to FAISS...")

index.add(embeddings)

print("Vectors in index:", index.ntotal)


# ---------------------------------------------------------
# Save index
# ---------------------------------------------------------

faiss.write_index(
    index,
    INDEX_FILE
)

print("\nFAISS index saved:")
print(INDEX_FILE)