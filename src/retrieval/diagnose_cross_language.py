import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


QUERIES = [
    "मैनहट्टन परियोजना क्या थी?",
    "What was the Manhattan Project?",
    "What was the purpose of the Manhattan Project?",
    "Who is the president of the United States?",
]


def main():

    print("=" * 80)
    print("CROSS-LANGUAGE RETRIEVAL DIAGNOSTIC")
    print("=" * 80)

    print("\nLoading retriever...")
    retriever = Retriever()

    print("\nLoading reranker...")
    reranker = Reranker()

    print("\nModels ready.")

    for query_index, query in enumerate(QUERIES, start=1):

        print("\n" + "=" * 80)
        print(f"QUERY {query_index}")
        print("=" * 80)

        print("Query:")
        print(query)

        # --------------------------------------------------
        # FAISS RETRIEVAL
        # --------------------------------------------------

        print("\nFAISS TOP-10")
        print("-" * 80)

        candidates = retriever.search(
            query,
            top_k=10
        )

        for rank, result in enumerate(candidates, start=1):

            print(
                f"\nRank {rank}"
            )

            print(
                f"FAISS score : {result.get('score', 0):.4f}"
            )

            print(
                f"Passage    : "
                f"{result.get('passage_id', 'N/A')}"
            )

            print(
                f"Text       : "
                f"{result.get('chunk', '')[:500]}"
            )

        # --------------------------------------------------
        # RERANKING
        # --------------------------------------------------

        print("\n" + "-" * 80)
        print("CROSS-ENCODER RERANKING")
        print("-" * 80)

        reranked = reranker.rerank(
            query,
            candidates,
            top_k=10
        )

        for rank, result in enumerate(reranked, start=1):

            print(
                f"\nRank {rank}"
            )

            print(
                f"Rerank score: "
                f"{result.get('rerank_score', result.get('score', 0)):.4f}"
            )

            print(
                f"FAISS score : "
                f"{result.get('score', 0):.4f}"
            )

            print(
                f"Passage     : "
                f"{result.get('passage_id', 'N/A')}"
            )

            print(
                f"Text        : "
                f"{result.get('chunk', '')[:500]}"
            )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()