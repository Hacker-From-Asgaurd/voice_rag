import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


def main():

    query = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"

    print("=" * 70)
    print("RERANKER TEST")
    print("=" * 70)

    retriever = Retriever()
    reranker = Reranker()

    print("\nRetrieving top 20 candidates...")

    results = retriever.search(
        query,
        top_k=20
    )

    print(
        f"Retrieved: {len(results)}"
    )

    print("\nReranking...")

    reranked = reranker.rerank(
        query,
        results,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    for i, result in enumerate(
        reranked,
        start=1
    ):

        print(
            f"\nResult #{i}"
        )

        print(
            f"FAISS score   : "
            f"{result['score']:.4f}"
        )

        print(
            f"Rerank score  : "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Selected      : "
            f"{result['is_selected']}"
        )

        print(
            f"Passage ID    : "
            f"{result['passage_id']}"
        )

        print(
            f"Chunk         : "
            f"{result['chunk'][:500]}"
        )


if __name__ == "__main__":
    main()