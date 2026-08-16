import json
import numpy as np
from sentence_transformers import SentenceTransformer


METADATA_FILE = "data/adaptive_metadata.json"

QUERY_ID = 1185869

OLD_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NEW_MODEL = "intfloat/multilingual-e5-base"


print("Loading metadata...")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


records = [
    r for r in metadata
    if r["query_id"] == QUERY_ID
]

if not records:
    raise ValueError("Query ID not found.")


query = records[0]["query"]

passages = [r["chunk"] for r in records]

labels = [r["is_selected"] for r in records]


print("\nQuery:")
print(query)

print("\nGround-truth labels:")
print(labels)


def test_model(model_name, query_prefix="", passage_prefix=""):

    print("\n" + "=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    model = SentenceTransformer(model_name)

    query_text = query_prefix + query

    passage_texts = [
        passage_prefix + passage
        for passage in passages
    ]

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True
    )[0]

    passage_embeddings = model.encode(
        passage_texts,
        normalize_embeddings=True
    )

    scores = np.dot(
        passage_embeddings,
        query_embedding
    )

    ranking = np.argsort(scores)[::-1]

    for rank, index in enumerate(ranking, start=1):

        print(
            f"\nRank {rank} | "
            f"Passage {index} | "
            f"Score {scores[index]:.4f} | "
            f"Selected {labels[index]}"
        )

        print(passages[index][:250])


# ---------------------------------------------------------
# Model 1
# ---------------------------------------------------------

test_model(
    OLD_MODEL
)


# ---------------------------------------------------------
# Model 2
# ---------------------------------------------------------

test_model(
    NEW_MODEL,
    query_prefix="query: ",
    passage_prefix="passage: "
)