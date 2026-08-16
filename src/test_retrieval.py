import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

INDEX_FILE = "data/adaptive.index"
METADATA_FILE = "data/adaptive_metadata.json"


print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)

print("Loading metadata...")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


query = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"


print("\nUser query:")
print(query)


# ---------------------------------------------------------
# Embed query
# ---------------------------------------------------------

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

k = 5

scores, indices = index.search(
    query_embedding,
    k
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP RETRIEVED RESULTS")
print("=" * 70)

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    record = metadata[idx]

    print(f"\nResult #{rank}")
    print("-" * 70)

    print("Similarity:", round(float(score), 4))
    print("Query ID:", record["query_id"])
    print("Passage ID:", record["passage_id"])
    print("Selected:", record["is_selected"])

    print("\nRetrieved chunk:")
    print(record["chunk"][:1000])