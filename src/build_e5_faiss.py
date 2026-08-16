import faiss
import numpy as np

EMBEDDINGS_FILE = "data/e5_embeddings.npy"
INDEX_FILE = "data/e5_adaptive.index"


print("Loading E5 embeddings...")

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
# Add embeddings
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

print("\nE5 FAISS index saved:")
print(INDEX_FILE)