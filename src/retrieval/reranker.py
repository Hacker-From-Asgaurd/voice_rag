from typing import List, Dict, Any, Tuple
import torch
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker:
    """
    Optimized CrossEncoder Reranker with Startup GPU Warmup,
    Batch Predict Acceleration, and Inference Mode.
    """

    def __init__(self):
        print("Loading multilingual reranker...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Reranker device:", self.device)

        self.model = CrossEncoder(
            MODEL_NAME,
            device=self.device
        )
        self.model.model.eval()

        self._warmup()
        print("Reranker ready (warmed).")

    def _warmup(self):
        """Warm up CUDA kernels during initialization."""
        try:
            with torch.inference_mode():
                dummy_pairs = [("warmup query", "warmup passage chunk")]
                self.model.predict(dummy_pairs, show_progress_bar=False)
            if self.device == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Reranker warmup notice: {e}")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        pairs: List[Tuple[str, str]] = [
            (query, result.get("chunk", ""))
            for result in results
        ]

        with torch.inference_mode():
            scores = self.model.predict(
                pairs,
                show_progress_bar=False
            )

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        # Sort descending by rerank score
        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return results[:top_k]