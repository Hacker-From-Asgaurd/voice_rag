import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-base"

INDEX_FILE = "data/e5_adaptive.index"
METADATA_FILE = "data/e5_metadata.json"


class Retriever:

    def __init__(self):

        print("Loading E5 model...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Using device:",
            self.device
        )

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=self.device
        )

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            INDEX_FILE
        )

        print("Loading metadata...")

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(f)

        print("Retriever ready.")

    def search(
        self,
        query,
        top_k=5
    ):

        query_embedding = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            record = self.metadata[idx]

            results.append({
                "score": float(score),
                "query_id": record["query_id"],
                "passage_id": record["passage_id"],
                "is_selected": record["is_selected"],
                "chunk": record["chunk"],
                "parent": record["parent"]
            })

        return results

    def get_context(
        self,
        query,
        top_k=5
    ):

        results = self.search(
            query,
            top_k=top_k
        )

        context_parts = []

        for i, result in enumerate(
            results,
            start=1
        ):

            context_parts.append(
                f"[Source {i}]\n"
                f"{result['chunk']}"
            )

        return "\n\n".join(
            context_parts
        )