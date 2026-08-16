import sys
from pathlib import Path

# Make src/ importable when this file is run directly.
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from harness.pipeline import RAGPipeline
from harness.schemas import QueryRequest


# Create ONE pipeline instance.
# The retriever/model/index are loaded once here.
_pipeline = RAGPipeline()


def answer_question(question, top_k=5):
    """
    Compatibility wrapper around the canonical RAGPipeline.

    All actual RAG processing is handled by harness.pipeline.RAGPipeline.
    """

    request = QueryRequest(
        query=question,
        top_k=top_k
    )

    response = _pipeline.run(request)

    return response.answer, response.sources


def main():

    question = input("\nAsk your question: ").strip()

    if not question:
        print("Query cannot be empty.")
        return

    answer, sources = answer_question(
        question,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print(answer)

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for i, source in enumerate(
        sources,
        start=1
    ):

        print(
            f"\nSource {i} "
            f"(similarity={source.score:.4f})"
        )

        print(
            source.chunk[:300]
        )


if __name__ == "__main__":
    main()