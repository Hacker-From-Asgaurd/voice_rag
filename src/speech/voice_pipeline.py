import time
from pathlib import Path
from typing import Optional, Union, BinaryIO

from harness.pipeline import (
    RAGPipeline,
    ABSTENTION_TEXT,
    MIN_RERANK_SCORE,
    RERANK_CANDIDATES,
    FINAL_RESULTS,
)
from harness.guardrails import check_input, UNSAFE_RESPONSE
from harness.schemas import (
    RetrievedSource,
    VoiceLatencyBreakdown,
    VoicePipelineResponse,
)
from speech.transcriber import SarvamTranscriber


class VoiceRAGPipeline:
    """
    End-to-End Voice RAG Pipeline:
    Audio Input -> Sarvam Saaras v3 STT -> Guardrails -> E5 Retrieval -> CrossEncoder -> Evidence Gate -> Gemini 3.5 Flash -> Grounded Answer.

    All models and clients are initialized ONCE.
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        transcriber: Optional[SarvamTranscriber] = None,
    ):
        print("Initializing Voice RAG Pipeline...")

        # Initialize speech transcriber (once)
        self.transcriber = transcriber or SarvamTranscriber()

        # Initialize RAG pipeline (once)
        self.rag = rag_pipeline or RAGPipeline()

        print("Voice RAG Pipeline ready.")

    def run(
        self,
        audio_source: Union[str, Path, bytes, BinaryIO],
        language_code: str = "unknown",
        filename: str = "audio.wav",
        top_k: int = FINAL_RESULTS,
    ) -> VoicePipelineResponse:
        """
        Process audio input end-to-end through STT and RAG.

        Args:
            audio_source: File path, bytes, or file-like buffer.
            language_code: Language hint or 'unknown' for automatic detection.
            filename: Audio filename for byte stream mime detection.
            top_k: Number of reranked results to retain for evidence.

        Returns:
            VoicePipelineResponse with transcript, grounded answer, sources, and latency breakdown.
        """
        t_e2e_start = time.perf_counter()
        lat = VoiceLatencyBreakdown()

        # -------------------------------------------------------------
        # 1. SPEECH-TO-TEXT (Sarvam AI Saaras v3)
        # -------------------------------------------------------------
        try:
            stt_result = self.transcriber.transcribe(
                audio_source=audio_source,
                language_code=language_code,
                filename=filename,
            )
            transcript = stt_result["transcript"]
            detected_lang = stt_result["language_code"]
            lat.stt_ms = stt_result["latency_ms"]
        except Exception as e:
            t_e2e_end = time.perf_counter()
            lat.total_ms = (t_e2e_end - t_e2e_start) * 1000.0
            return VoicePipelineResponse(
                success=False,
                transcript="",
                language_code=language_code,
                answer="आवाज़ को पाठ में बदलने में त्रुटि हुई। कृपया पुनः प्रयास करें।",
                grounded=False,
                sources=[],
                latency=lat,
                error=f"STT Error: {type(e).__name__}",
            )

        # Handle empty/silent audio transcript
        if not transcript or not transcript.strip():
            t_e2e_end = time.perf_counter()
            lat.total_ms = (t_e2e_end - t_e2e_start) * 1000.0
            return VoicePipelineResponse(
                success=False,
                transcript="",
                language_code=detected_lang,
                answer="कोई स्पष्ट आवाज़ या प्रश्न प्राप्त नहीं हुआ। कृपया पुनः बोलें।",
                grounded=False,
                sources=[],
                latency=lat,
                error="Empty transcript from audio.",
            )

        query = transcript.strip()

        # -------------------------------------------------------------
        # 2. INPUT GUARDRAIL
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        guardrail_result = check_input(query)
        t1 = time.perf_counter()
        lat.guardrail_ms = (t1 - t0) * 1000.0

        if not guardrail_result["allowed"]:
            t_e2e_end = time.perf_counter()
            lat.total_ms = (t_e2e_end - t_e2e_start) * 1000.0
            return VoicePipelineResponse(
                success=True,
                transcript=query,
                language_code=detected_lang,
                answer=UNSAFE_RESPONSE,
                grounded=False,
                sources=[],
                latency=lat,
                error=None,
            )

        # -------------------------------------------------------------
        # 3. E5 DENSE RETRIEVAL (k=15 Candidates)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            candidates = self.rag.retriever.search(query, top_k=RERANK_CANDIDATES)
        except Exception as e:
            candidates = []
        t1 = time.perf_counter()
        lat.retrieval_ms = (t1 - t0) * 1000.0

        if not candidates:
            t_e2e_end = time.perf_counter()
            lat.total_ms = (t_e2e_end - t_e2e_start) * 1000.0
            return VoicePipelineResponse(
                success=True,
                transcript=query,
                language_code=detected_lang,
                answer=ABSTENTION_TEXT,
                grounded=False,
                sources=[],
                latency=lat,
                error=None,
            )

        # -------------------------------------------------------------
        # 4. CROSS-ENCODER RERANKING (Top-5)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            reranked = self.rag.reranker.rerank(query, candidates, top_k=top_k)
        except Exception as e:
            reranked = candidates[:top_k]
        t1 = time.perf_counter()
        lat.reranker_ms = (t1 - t0) * 1000.0

        # -------------------------------------------------------------
        # 5. EVIDENCE GATE (Score Filtering)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        evidence = self.rag.filter_evidence(reranked)
        t1 = time.perf_counter()
        lat.evidence_gate_ms = (t1 - t0) * 1000.0

        # -------------------------------------------------------------
        # 6. GENERATION / ABSTENTION
        # -------------------------------------------------------------
        answer = ""
        grounded = False

        if evidence:
            context = self.rag.build_context(evidence)
            t0 = time.perf_counter()
            try:
                answer = self.rag.generate(query, context)
            except Exception as e:
                answer = ABSTENTION_TEXT
            t1 = time.perf_counter()
            lat.generation_ms = (t1 - t0) * 1000.0

            if answer.strip() != ABSTENTION_TEXT and "उपलब्ध नहीं" not in answer:
                grounded = True
            else:
                grounded = False
        else:
            answer = ABSTENTION_TEXT
            grounded = False

        # Format structured sources
        structured_sources = [
            RetrievedSource(
                score=float(r.get("rerank_score", r.get("score", 0.0))),
                chunk=r.get("chunk", ""),
                query_id=int(r.get("query_id", 0)),
                passage_id=int(r.get("passage_id", 0)),
                is_selected=int(r.get("is_selected", 0)),
            )
            for r in (evidence if evidence else reranked)
        ]

        t_e2e_end = time.perf_counter()
        lat.total_ms = (t_e2e_end - t_e2e_start) * 1000.0

        return VoicePipelineResponse(
            success=True,
            transcript=query,
            language_code=detected_lang,
            answer=answer,
            grounded=grounded,
            sources=structured_sources,
            latency=lat,
            error=None,
        )
