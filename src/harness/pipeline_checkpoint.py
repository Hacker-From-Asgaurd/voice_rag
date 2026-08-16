import sys
from pathlib import Path

# Add src/ to Python import path
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from retrieval.retriever import Retriever
from generation.generator import generate_answer

from harness.schemas import (
    QueryRequest,
    PipelineResponse,
    RetrievedSource
)

from harness.retry import retry


class RAGPipeline:

    def __init__(self):

        print("Initializing RAG pipeline...")

        self.retriever = Retriever()

        print("RAG pipeline ready.")

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    def validate_request(self, request):

        if not isinstance(request.query, str):
            raise ValueError("Query must be a string.")

        query = request.query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if len(query) > 1000:
            raise ValueError(
                "Query is too long. Maximum length is 1000 characters."
            )

        if request.top_k < 1 or request.top_k > 20:
            raise ValueError(
                "top_k must be between 1 and 20."
            )

    # ---------------------------------------------------------
    # RETRIEVAL WITH RETRY
    # ---------------------------------------------------------

    @retry(max_attempts=2, delay=0.5)
    def retrieve(self, request):

        return self.retriever.search(
            request.query,
            top_k=request.top_k
        )

    # ---------------------------------------------------------
    # GENERATION WITH RETRY
    # ---------------------------------------------------------

    @retry(max_attempts=2, delay=0.5)
    def generate(self, query, context):

        return generate_answer(
            query,
            context
        )

    # ---------------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------------

    def run(self, request):

        try:

            # 1. Validate input
            self.validate_request(request)

            print("Retrieving context...")

            # 2. Retrieve
            results = self.retrieve(request)

            if not results:

                return PipelineResponse(
                    success=False,
                    query=request.query,
                    answer=(
                        "मुझे इस प्रश्न का उत्तर देने के लिए "
                        "पर्याप्त जानकारी नहीं मिली।"
                    ),
                    sources=[],
                    grounded=False,
                    error="No retrieval results."
                )

            # 3. Build context
            context_parts = []
            sources = []

            for result in results:

                context_parts.append(
                    f"[Source]\n{result['chunk']}"
                )

                sources.append(
                    RetrievedSource(
                        score=result["score"],
                        chunk=result["chunk"],
                        query_id=result["query_id"],
                        passage_id=result["passage_id"],
                        is_selected=result["is_selected"]
                    )
                )

            context = "\n\n".join(context_parts)

            print("Context retrieved.")
            print("Generating answer...")

            # 4. Generate answer
            answer = self.generate(
                request.query,
                context
            )

            # 5. Return structured response
            return PipelineResponse(
                success=True,
                query=request.query,
                answer=answer,
                sources=sources,
                grounded=True
            )

        except Exception as e:

            print(f"[Pipeline Error] {e}")

            return PipelineResponse(
                success=False,
                query=request.query,
                answer=(
                    "क्षमा करें, इस समय उत्तर उत्पन्न "
                    "नहीं किया जा सका।"
                ),
                sources=[],
                grounded=False,
                error=str(e)
            )