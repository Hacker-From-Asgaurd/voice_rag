from speech.transcriber import transcribe_audio

from harness.pipeline import RAGPipeline
from harness.schemas import QueryRequest


# Load the RAG system once.
pipeline = RAGPipeline()


def voice_rag(
    audio_path,
    language_code="en-IN"
):
    """
    Complete Voice RAG pipeline.

    Audio
      ↓
    Sarvam AI STT
      ↓
    Guardrails
      ↓
    E5 + FAISS retrieval
      ↓
    CrossEncoder reranking
      ↓
    Evidence filtering
      ↓
    Gemini grounded generation
    """

    print("\n" + "=" * 70)
    print("STEP 1 — TRANSCRIPTION")
    print("=" * 70)

    question = transcribe_audio(
        audio_path,
        language_code=language_code
    )

    if not question or not question.strip():

        raise ValueError(
            "Speech transcription returned an empty result."
        )

    question = question.strip()

    print("\nTranscribed question:")
    print(question)

    print("\n" + "=" * 70)
    print("STEP 2 — RAG PIPELINE")
    print("=" * 70)

    request = QueryRequest(
        query=question,
        top_k=5
    )

    response = pipeline.run(request)

    print("\n" + "=" * 70)
    print("FINAL RESPONSE")
    print("=" * 70)

    print("Success :", response.success)
    print("Grounded:", response.grounded)

    print("\nAnswer:")
    print(response.answer)

    print(
        f"\nSources: {len(response.sources)}"
    )

    return (
        question,
        response.answer,
        response.sources
    )


if __name__ == "__main__":

    audio_path = "test_audio.wav"

    question, answer, sources = voice_rag(
        audio_path,
        language_code="en-IN"
    )

    print("\n" + "=" * 70)
    print("VOICE RAG COMPLETE")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for i, source in enumerate(
        sources,
        start=1
    ):

        print(
            f"\nSource {i} "
            f"(score={source.score:.4f})"
        )

        print(
            source.chunk[:300]
        )