import time
import uuid
from pathlib import Path
from typing import Optional, Union, BinaryIO

from harness.pipeline import (
    RAGPipeline,
    ABSTENTION_TEXT,
    EVIDENCE_RELEVANCE_THRESHOLD,
    RERANK_CANDIDATES,
    FINAL_RESULTS,
)
from harness.guardrails import (
    check_actionable_safety,
    calibrate_crossencoder_score,
    verify_grounding,
    UNSAFE_RESPONSE,
)
from harness.schemas import (
    RetrievedSource,
    SpeechToTextResult,
    GuardrailResult,
    DenseRetrievalResult,
    RerankingResult,
    EvidenceGateResult,
    GenerationResult,
    VoiceLatencyBreakdown,
    RAGPipelineTrace,
    VoicePipelineResponse,
)
from speech.transcriber import SarvamTranscriber


class VoiceRAGPipeline:
    """
    End-to-End Voice RAG Pipeline with Complete Typed Trace Orchestration.
    Audio Input -> Sarvam Saaras v3 STT -> Actionable Safety Gate -> E5 Retrieval -> CrossEncoder -> Platt Calibration Gate -> Gemini Flash -> Grounding Check.
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        transcriber: Optional[SarvamTranscriber] = None,
    ):
        print("Initializing Voice RAG Pipeline...")
        self.transcriber = transcriber or SarvamTranscriber()
        self.rag = rag_pipeline or RAGPipeline()
        print("Voice RAG Pipeline ready.")

    def run(
        self,
        audio_source: Union[str, Path, bytes, BinaryIO],
        language_code: str = "unknown",
        filename: str = "audio.wav",
        top_k: int = FINAL_RESULTS,
    ) -> VoicePipelineResponse:
        trace_id = str(uuid.uuid4())
        trace = RAGPipelineTrace(trace_id=trace_id, timestamp=time.time())
        t_e2e_start = time.perf_counter()
        lat = VoiceLatencyBreakdown()

        # -------------------------------------------------------------
        # 1. SPEECH-TO-TEXT (Sarvam Saaras v3)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            stt_res = self.transcriber.transcribe(
                audio_source,
                language_code=language_code,
                filename=filename,
            )
            query = stt_res.get("transcript", "").strip()
            detected_lang = stt_res.get("language_code", language_code)
            stt_dur = (time.perf_counter() - t0) * 1000.0
            lat.stt_ms = stt_dur
            trace.stt = SpeechToTextResult(
                transcript=query,
                language_code=detected_lang,
                duration_ms=stt_dur,
                success=True,
            )
        except Exception as e:
            stt_dur = (time.perf_counter() - t0) * 1000.0
            lat.stt_ms = stt_dur
            lat.total_ms = (time.perf_counter() - t_e2e_start) * 1000.0
            trace.stt = SpeechToTextResult(
                transcript="",
                language_code=language_code,
                duration_ms=stt_dur,
                success=False,
                error=str(e),
            )
            return VoicePipelineResponse(
                success=False,
                transcript="",
                language_code=language_code,
                answer="Speech recognition failed. Please try speaking again.",
                grounded=False,
                sources=[],
                latency=lat,
                trace_id=trace_id,
                error=f"STT Error: {e}",
                trace=trace,
            )

        trace.query = query
        trace.language = detected_lang

        if not query:
            lat.total_ms = (time.perf_counter() - t_e2e_start) * 1000.0
            return VoicePipelineResponse(
                success=True,
                transcript="",
                language_code=detected_lang,
                answer="No clear speech detected in audio.",
                grounded=False,
                sources=[],
                latency=lat,
                trace_id=trace_id,
                error="Empty transcription.",
                trace=trace,
            )

        # -------------------------------------------------------------
        # 2. ACTIONABLE SAFETY GUARDRAIL
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        guard = check_actionable_safety(query)
        guard_dur = (time.perf_counter() - t0) * 1000.0
        lat.guardrail_ms = guard_dur
        trace.guardrail = GuardrailResult(
            allowed=guard["allowed"],
            reason=guard["reason"],
            category=guard.get("category", "safe"),
            duration_ms=guard_dur,
        )

        if not guard["allowed"]:
            lat.total_ms = (time.perf_counter() - t_e2e_start) * 1000.0
            trace.final_status = "BLOCKED_SAFETY"
            return VoicePipelineResponse(
                success=True,
                transcript=query,
                language_code=detected_lang,
                answer=UNSAFE_RESPONSE,
                grounded=False,
                sources=[],
                latency=lat,
                trace_id=trace_id,
                error=guard["reason"],
                trace=trace,
            )

        # -------------------------------------------------------------
        # 3. DENSE RETRIEVAL (E5 + FAISS)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        candidates = self.rag.retriever.search(query, top_k=RERANK_CANDIDATES)
        ret_dur = (time.perf_counter() - t0) * 1000.0
        lat.retrieval_ms = ret_dur
        trace.retrieval = DenseRetrievalResult(
            candidates=candidates,
            candidate_count=len(candidates),
            duration_ms=ret_dur,
        )

        if not candidates:
            lat.total_ms = (time.perf_counter() - t_e2e_start) * 1000.0
            trace.final_status = "ABSTAIN_NO_CANDIDATES"
            return VoicePipelineResponse(
                success=True,
                transcript=query,
                language_code=detected_lang,
                answer=ABSTENTION_TEXT,
                grounded=False,
                sources=[],
                latency=lat,
                trace_id=trace_id,
                error=None,
                trace=trace,
            )

        # -------------------------------------------------------------
        # 4. CROSS-ENCODER RERANKING
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        reranked = self.rag.reranker.rerank(query, candidates, top_k=top_k)
        rerank_dur = (time.perf_counter() - t0) * 1000.0
        lat.reranker_ms = rerank_dur
        trace.reranking = RerankingResult(
            reranked=reranked,
            top_k_count=len(reranked),
            duration_ms=rerank_dur,
        )

        # -------------------------------------------------------------
        # 5. EVIDENCE CALIBRATION GATE (Platt Scaling)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        evidence = self.rag.filter_evidence(reranked)
        gate_dur = (time.perf_counter() - t0) * 1000.0
        lat.evidence_gate_ms = gate_dur

        top_raw = float(reranked[0].get("rerank_score", 0.0)) if reranked else 0.0
        top_calib = float(reranked[0].get("calibrated_confidence", calibrate_crossencoder_score(top_raw))) if reranked else 0.0

        trace.evidence_gate = EvidenceGateResult(
            passed=bool(evidence),
            top_calibrated_confidence=top_calib,
            top_raw_score=top_raw,
            threshold=EVIDENCE_RELEVANCE_THRESHOLD,
            retained_evidence=evidence,
            duration_ms=gate_dur,
        )

        # -------------------------------------------------------------
        # 6. GROUNDED GENERATION & POST-GEN VERIFICATION
        # -------------------------------------------------------------
        if evidence:
            context = self.rag.build_context(evidence)
            t0 = time.perf_counter()
            try:
                answer = self.rag.generate(query, context)
            except Exception as e:
                answer = ABSTENTION_TEXT
            gen_dur = (time.perf_counter() - t0) * 1000.0
            lat.generation_ms = gen_dur
            grounded = verify_grounding(answer, context, min_confidence=EVIDENCE_RELEVANCE_THRESHOLD)
            trace.generation = GenerationResult(
                answer=answer,
                grounded=grounded,
                grounding_status="GROUNDED" if grounded else "ABSTAINED",
                duration_ms=gen_dur,
            )
        else:
            answer = ABSTENTION_TEXT
            grounded = False
            trace.generation = GenerationResult(
                answer=ABSTENTION_TEXT,
                grounded=False,
                grounding_status="INSUFFICIENT_EVIDENCE",
                duration_ms=0.0,
            )

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

        lat.total_ms = (time.perf_counter() - t_e2e_start) * 1000.0
        trace.latency = lat
        trace.final_status = "GROUNDED_SUCCESS" if grounded else "SAFE_ABSTENTION"

        return VoicePipelineResponse(
            success=True,
            transcript=query,
            language_code=detected_lang,
            answer=answer,
            grounded=grounded,
            sources=structured_sources,
            latency=lat,
            trace_id=trace_id,
            error=None,
            trace=trace,
        )

    def query_text(self, text: str, top_k: int = FINAL_RESULTS) -> VoicePipelineResponse:
        """
        Direct text query execution bypassing STT for fast text testing.
        """
        trace_id = str(uuid.uuid4())
        trace = RAGPipelineTrace(trace_id=trace_id, timestamp=time.time(), query=text)
        t_e2e_start = time.perf_counter()
        lat = VoiceLatencyBreakdown()
        lat.stt_ms = 0.0

        # 1. Actionable Safety
        t0 = time.perf_counter()
        guard = check_actionable_safety(text)
        lat.guardrail_ms = (time.perf_counter() - t0) * 1000.0

        if not guard["allowed"]:
            lat.total_ms = (time.perf_counter() - t_e2e_start) * 1000.0
            return VoicePipelineResponse(
                success=True,
                transcript=text,
                language_code="unknown",
                answer=UNSAFE_RESPONSE,
                grounded=False,
                sources=[],
                latency=lat,
                trace_id=trace_id,
                error=guard["reason"],
                trace=trace,
            )

        # 2. Dense Retrieval
        t0 = time.perf_counter()
        candidates = self.rag.retriever.search(text, top_k=RERANK_CANDIDATES)
        lat.retrieval_ms = (time.perf_counter() - t0) * 1000.0

        if not candidates:
            lat.total_ms = (time.perf_counter() - t_e2e_start) * 1000.0
            return VoicePipelineResponse(
                success=True,
                transcript=text,
                language_code="unknown",
                answer=ABSTENTION_TEXT,
                grounded=False,
                sources=[],
                latency=lat,
                trace_id=trace_id,
                error=None,
                trace=trace,
            )

        # 3. CrossEncoder Reranking
        t0 = time.perf_counter()
        reranked = self.rag.reranker.rerank(text, candidates, top_k=top_k)
        lat.reranker_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Evidence Gate
        t0 = time.perf_counter()
        evidence = self.rag.filter_evidence(reranked)
        lat.evidence_gate_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Generation & Grounding Check
        if evidence:
            context = self.rag.build_context(evidence)
            t0 = time.perf_counter()
            try:
                answer = self.rag.generate(text, context)
            except Exception as e:
                answer = ABSTENTION_TEXT
            lat.generation_ms = (time.perf_counter() - t0) * 1000.0
            grounded = verify_grounding(answer, context, min_confidence=EVIDENCE_RELEVANCE_THRESHOLD)
        else:
            answer = ABSTENTION_TEXT
            grounded = False

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

        lat.total_ms = (time.perf_counter() - t_e2e_start) * 1000.0
        return VoicePipelineResponse(
            success=True,
            transcript=text,
            language_code="unknown",
            answer=answer,
            grounded=grounded,
            sources=structured_sources,
            latency=lat,
            trace_id=trace_id,
            error=None,
            trace=trace,
        )
