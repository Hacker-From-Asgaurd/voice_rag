import sys
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

# Make src/ importable
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.generator import generate_answer
from harness.schemas import (
    QueryRequest,
    PipelineResponse,
    RetrievedSource,
    GuardrailResult,
    DenseRetrievalResult,
    RerankingResult,
    EvidenceGateResult,
    GenerationResult,
    VoiceLatencyBreakdown,
    RAGPipelineTrace,
)
from harness.retry import retry
from harness.guardrails import (
    check_actionable_safety,
    calibrate_crossencoder_score,
    verify_grounding,
    UNSAFE_RESPONSE,
    ABSTENTION_TEXT,
)

# Calibrated Evidence Threshold (Platt-scaled relevance probability >= 0.50 corresponds to positive CrossEncoder score)
EVIDENCE_RELEVANCE_THRESHOLD = 0.50

# Retrieval & Reranker Budgets
RERANK_CANDIDATES = 15
FINAL_RESULTS = 5


class RAGPipeline:
    """
    Production-grade Grounded RAG Pipeline with End-to-End Typed Trace Orchestration,
    Fitted Platt Relevance Scaling, and Post-Generation Grounding Verification.
    """

    def __init__(self):
        print("Initializing RAG pipeline (E5 Dense Retriever + mMARCO CrossEncoder)...")
        # Models and FAISS index are loaded ONCE
        self.retriever = Retriever()
        self.reranker = Reranker()
        print("RAG pipeline ready.")

    def validate_request(self, request: QueryRequest):
        if not isinstance(request.query, str):
            raise ValueError("Query must be a string.")
        query = request.query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if len(query) > 1000:
            raise ValueError("Query is too long. Maximum length is 1000 characters.")
        if request.top_k < 1 or request.top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")

    def retrieve(self, request: QueryRequest) -> List[Dict[str, Any]]:
        candidates = self.retriever.search(request.query, top_k=RERANK_CANDIDATES)
        if not candidates:
            return []
        reranked = self.reranker.rerank(request.query, candidates, top_k=FINAL_RESULTS)
        return reranked

    @retry(max_attempts=2, delay=0.5)
    def generate(self, query: str, context: str) -> str:
        return generate_answer(query, context)

    def build_context(self, results: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, result in enumerate(results, start=1):
            context_parts.append(f"[Source {i}]\n{result.get('chunk', '')}")
        return "\n\n".join(context_parts)

    def filter_evidence(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters evidence using fitted Platt scaling relevance calibration.
        Threshold T=0.80 is applied to calibrated relevance probability.
        """
        filtered = []
        for r in results:
            raw_score = float(r.get("rerank_score", r.get("score", 0.0)))
            calibrated_prob = calibrate_crossencoder_score(raw_score)
            r["calibrated_confidence"] = calibrated_prob
            if calibrated_prob >= EVIDENCE_RELEVANCE_THRESHOLD:
                filtered.append(r)
        return filtered

    def run(self, request: QueryRequest) -> PipelineResponse:
        trace_id = str(uuid.uuid4())
        t_start = time.perf_counter()
        lat = VoiceLatencyBreakdown()

        # 1. Validation
        self.validate_request(request)

        # 2. Input Safety Guardrail
        t0 = time.perf_counter()
        guard = check_actionable_safety(request.query)
        lat.guardrail_ms = (time.perf_counter() - t0) * 1000.0

        if not guard["allowed"]:
            return PipelineResponse(
                success=True,
                query=request.query,
                answer=UNSAFE_RESPONSE,
                sources=[],
                grounded=False,
                trace_id=trace_id,
                latency=lat,
                error=guard["reason"],
            )

        # 3. Dense Retrieval (E5)
        t0 = time.perf_counter()
        candidates = self.retriever.search(request.query, top_k=RERANK_CANDIDATES)
        lat.retrieval_ms = (time.perf_counter() - t0) * 1000.0

        if not candidates:
            lat.total_ms = (time.perf_counter() - t_start) * 1000.0
            return PipelineResponse(
                success=True,
                query=request.query,
                answer=ABSTENTION_TEXT,
                sources=[],
                grounded=False,
                trace_id=trace_id,
                latency=lat,
                error="No vector candidates retrieved.",
            )

        # 4. CrossEncoder Reranking (Top-5)
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(request.query, candidates, top_k=FINAL_RESULTS)
        lat.reranker_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Calibrated Evidence Relevance Gate
        t0 = time.perf_counter()
        evidence = self.filter_evidence(reranked)
        lat.evidence_gate_ms = (time.perf_counter() - t0) * 1000.0

        # 6. Generation & Post-Generation Grounding Verification
        if evidence:
            context = self.build_context(evidence)
            t0 = time.perf_counter()
            try:
                answer = self.generate(request.query, context)
            except Exception as e:
                answer = ABSTENTION_TEXT
            lat.generation_ms = (time.perf_counter() - t0) * 1000.0
            grounded = verify_grounding(answer, context, min_confidence=EVIDENCE_RELEVANCE_THRESHOLD)
        else:
            answer = ABSTENTION_TEXT
            grounded = False

        # Format structured sources
        structured_sources = [
            RetrievedSource(
                score=float(r.get("rerank_score", r.get("score", 0.0))),
                calibrated_confidence=float(r.get("calibrated_confidence", calibrate_crossencoder_score(float(r.get("rerank_score", r.get("score", 0.0)))))),
                chunk=r.get("chunk", ""),
                query_id=int(r.get("query_id", 0)),
                passage_id=int(r.get("passage_id", 0)),
                parent_passage_id=int(r.get("parent_passage_id", r.get("passage_id", 0))),
                is_selected=int(r.get("is_selected", 0)),
            )
            for r in (evidence if evidence else reranked)
        ]

        lat.total_ms = (time.perf_counter() - t_start) * 1000.0

        return PipelineResponse(
            success=True,
            query=request.query,
            answer=answer,
            sources=structured_sources,
            grounded=grounded,
            trace_id=trace_id,
            latency=lat,
            error=None,
        )