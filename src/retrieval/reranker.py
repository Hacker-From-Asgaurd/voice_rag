import torch
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker:

    def __init__(self):

        print("Loading multilingual reranker...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Reranker device:",
            self.device
        )

        self.model = CrossEncoder(
            MODEL_NAME,
            device=self.device
        )

        print("Reranker ready.")

    def rerank(
        self,
        query,
        results,
        top_k=10
    ):

        if not results:
            return []

        pairs = []

        for result in results:

            pairs.append(
                (
                    query,
                    result["chunk"]
                )
            )

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for result, score in zip(
            results,
            scores
        ):

            result_copy = result.copy()

            result_copy["rerank_score"] = float(
                score
            )

            reranked.append(
                result_copy
            )

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]