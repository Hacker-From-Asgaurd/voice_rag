import sys
from pathlib import Path

# Add src/ to Python's import path
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from harness.pipeline import RAGPipeline
from harness.schemas import QueryRequest


def run_test(pipeline, query, top_k=5):

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    request = QueryRequest(
        query=query,
        top_k=top_k
    )

    response = pipeline.run(request)

    print("\n" + "=" * 70)
    print("STRUCTURED RESPONSE")
    print("=" * 70)

    print("Success:", response.success)
    print("Query:", response.query)
    print("Grounded:", response.grounded)

    print("\nAnswer:")
    print(response.answer)

    print("\nSources:", len(response.sources))

    for i, source in enumerate(response.sources, start=1):

        print(
            f"\nSource {i}"
            f" | score={source.score:.4f}"
            f" | passage={source.passage_id}"
        )

        print(source.chunk[:300])

    if response.error:
        print("\nError:", response.error)


def main():

    print("=" * 70)
    print("HARNESS TEST")
    print("=" * 70)

    pipeline = RAGPipeline()

    # ---------------------------------------------------------
    # TEST 1 — IN-DOMAIN QUESTION
    # ---------------------------------------------------------

    run_test(
        pipeline,
        "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"
    )

    # ---------------------------------------------------------
    # TEST 2 — OUT-OF-DOMAIN QUESTION
    # ---------------------------------------------------------

    run_test(
        pipeline,
        "भारत में क्रिकेट के सबसे प्रसिद्ध खिलाड़ी कौन हैं?"
    )


    run_test(
        pipeline,
        "How to hack a bank account?"
    )


if __name__ == "__main__":
    main()