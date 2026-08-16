import json
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
INDEX_FILE = "data/e5_adaptive.index"
METADATA_FILE = "data/e5_metadata.json"


class Retriever:
    """
    Optimized Multilingual E5 Dense Retriever with Startup Warmup,
    Zero-Allocation Query Formatting, and Inference Mode Acceleration.
    """

    def __init__(self):
        print("Loading E5 model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=self.device
        )
        self.model.eval()

        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_FILE)

        print("Loading metadata...")
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Warmup GPU kernel on startup to eliminate live cold start
        self._warmup()
        print("Retriever ready (warmed).")

    def _warmup(self):
        """Warm up CUDA kernels during initialization."""
        try:
            with torch.inference_mode():
                dummy_emb = self.model.encode(
                    ["query: warmup"],
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                dummy_np = np.asarray(dummy_emb, dtype="float32")
                self.index.search(dummy_np, 5)
            if self.device == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Retriever warmup notice: {e}")

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        prefixed_query = "query: " + query

        if torch.cuda.is_available() and getattr(self.model, "device", None) is not None and self.model.device.type != "cuda":
            try:
                self.model.to("cuda")
            except Exception:
                pass

        with torch.inference_mode():
            query_embedding = self.model.encode(
                [prefixed_query],
                normalize_embeddings=True,
                show_progress_bar=False
            )

        query_embedding_np = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding_np,
            top_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            record = self.metadata[idx]
            results.append({
                "score": float(score),
                "query_id": record.get("query_id", 0),
                "passage_id": record.get("passage_id", idx),
                "parent_passage_id": record.get("parent_passage_id", record.get("passage_id", idx)),
                "is_selected": record.get("is_selected", 0),
                "chunk": record.get("chunk", ""),
                "parent": record.get("parent", "")
            })

        return results

    def get_context(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k)
        return "\n\n".join([r["chunk"] for r in results])