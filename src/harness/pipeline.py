import sys
from pathlib import Path

# ---------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.generator import generate_answer

from harness.schemas import (
    QueryRequest,
    PipelineResponse,
    RetrievedSource
)

from harness.retry import retry
from harness.guardrails import check_input, UNSAFE_RESPONSE


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ABSTENTION_TEXT = (
    "जानकारी दिए गए संदर्भ में उपलब्ध नहीं है।"
)

# Keep this at 0.85 for now.
# We will NOT change it until final end-to-end validation.
MIN_RERANK_SCORE = 0.80

# Experimentally validated candidate size from Phase 3.
RERANK_CANDIDATES = 15

# Number of results sent to generation.
FINAL_RESULTS = 5


# ---------------------------------------------------------
# RAG PIPELINE
# ---------------------------------------------------------

class RAGPipeline:

    def __init__(self):

        print("Initializing RAG pipeline...")

        # Models and FAISS index are loaded ONCE.
        self.retriever = Retriever()
        self.reranker = Reranker()

        print("RAG pipeline ready.")

    # -----------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------

    def validate_request(self, request):

        if not isinstance(request.query, str):
            raise ValueError(
                "Query must be a string."
            )

        query = request.query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if len(query) > 1000:
            raise ValueError(
                "Query is too long. "
                "Maximum length is 1000 characters."
            )

        if request.top_k < 1 or request.top_k > 20:
            raise ValueError(
                "top_k must be between 1 and 20."
            )

    # -----------------------------------------------------
    # RETRIEVAL + RERANKING
    # -----------------------------------------------------

    def retrieve(self, request):

        # Stage 1:
        # E5 + FAISS retrieves experimentally validated
        # Top-10 candidates.
        candidates = self.retriever.search(
            request.query,
            top_k=RERANK_CANDIDATES
        )

        if not candidates:
            return []

        # Stage 2:
        # CrossEncoder reranks those candidates.
        reranked = self.reranker.rerank(
            request.query,
            candidates,
            top_k=FINAL_RESULTS
        )

        return reranked

    # -----------------------------------------------------
    # GENERATION
    # -----------------------------------------------------

    @retry(
        max_attempts=2,
        delay=0.5
    )
    def generate(
        self,
        query,
        context
    ):

        return generate_answer(
            query,
            context
        )

    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    def build_context(
        self,
        results
    ):

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

    # -----------------------------------------------------
    # EVIDENCE FILTER
    # -----------------------------------------------------

    def filter_evidence(
        self,
        results
    ):

        return [
            result
            for result in results
            if float(result.get("rerank_score", float("-inf")))
            >= MIN_RERANK_SCORE
        ]

    # -----------------------------------------------------
    # STRUCTURED SOURCES
    # -----------------------------------------------------

    def build_sources(
        self,
        evidence
    ):

        sources = []

        for result in evidence:

            sources.append(
                RetrievedSource(
                    score=float(
                        result["score"]
                    ),
                    chunk=result["chunk"],
                    query_id=int(
                        result["query_id"]
                    ),
                    passage_id=int(
                        result["passage_id"]
                    ),
                    is_selected=int(
                        result["is_selected"]
                    )
                )
            )

        return sources

    # -----------------------------------------------------
    # MAIN PIPELINE
    # -----------------------------------------------------

    def run(
        self,
        request
    ):

        try:

            # =============================================
            # 1. VALIDATE
            # =============================================

            self.validate_request(
                request
            )


            # =============================================
            # 1.5 SAFETY GUARDRAIL
            # =============================================

            guardrail_result = check_input(
            request.query
            )

            if not guardrail_result["allowed"]:

                print(
                    "Guardrail blocked query."
                )

                return PipelineResponse(
                    success=True,
                    query=request.query,
                    answer=UNSAFE_RESPONSE,
                    sources=[],
                    grounded=False,
                    error=guardrail_result["reason"]
                 )

            print(
                "Retrieving context..."
            )

            # =============================================
            # 2. RETRIEVE + RERANK
            # =============================================

            results = self.retrieve(
                request
            )

            print(
                f"Retrieved "
                f"{len(results)} reranked results."
            )

            if not results:

                return PipelineResponse(
                    success=False,
                    query=request.query,
                    answer=ABSTENTION_TEXT,
                    sources=[],
                    grounded=False,
                    error="No retrieval results."
                )

            # =============================================
            # 3. EVIDENCE GATE
            # =============================================

            evidence = self.filter_evidence(
                results
            )

            print(
                f"Evidence candidates: "
                f"{len(evidence)}"
            )

            # =============================================
            # 4. ABSTAIN
            # =============================================

            if not evidence:

                print(
                    "No sufficiently strong evidence. "
                    "Abstaining."
                )

                return PipelineResponse(
                    success=True,
                    query=request.query,
                    answer=ABSTENTION_TEXT,
                    sources=[],
                    grounded=False,
                    error=(
                        "Insufficient retrieval evidence."
                    )
                )

            # =============================================
            # 5. STRUCTURED SOURCES
            # =============================================

            sources = self.build_sources(
                evidence
            )

            # =============================================
            # 6. BUILD CONTEXT
            # =============================================

            context = self.build_context(
                evidence
            )

            print(
                "Context retrieved."
            )

            # =============================================
            # 7. GENERATE
            # =============================================

            print(
                "Generating answer..."
            )

            answer = self.generate(
                request.query,
                context
            )

            # =============================================
            # 8. OUTPUT VALIDATION
            # =============================================

            if not answer or not answer.strip():

                return PipelineResponse(
                    success=False,
                    query=request.query,
                    answer=(
                        "The system could not "
                        "generate an answer."
                    ),
                    sources=sources,
                    grounded=False,
                    error=(
                        "Generator returned "
                        "an empty answer."
                    )
                )

            answer = answer.strip()

            is_abstention = (
                answer == ABSTENTION_TEXT
            )

            # =============================================
            # 9. FINAL RESPONSE
            # =============================================

            return PipelineResponse(
                success=True,
                query=request.query,
                answer=answer,
                sources=sources,
                grounded=not is_abstention
            )

        except Exception as e:

            print(
                f"[Pipeline Error] {e}"
            )

            return PipelineResponse(
                success=False,
                query=request.query,
                answer=(
                    "The system could not "
                    "process the request."
                ),
                sources=[],
                grounded=False,
                error=str(e)
            )