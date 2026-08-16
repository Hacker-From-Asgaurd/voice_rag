import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class QueryRequest:
    query: str
    top_k: int = 5


@dataclass
class RetrievedSource:
    score: float
    calibrated_confidence: float = 0.0
    chunk: str = ""
    query_id: int = 0
    passage_id: int = 0
    parent_passage_id: int = 0
    is_selected: int = 0


@dataclass
class SpeechToTextResult:
    transcript: str = ""
    language_code: str = "unknown"
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class GuardrailResult:
    allowed: bool = True
    reason: Optional[str] = None
    category: str = "safe"
    duration_ms: float = 0.0


@dataclass
class DenseRetrievalResult:
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0
    duration_ms: float = 0.0


@dataclass
class RerankingResult:
    reranked: List[Dict[str, Any]] = field(default_factory=list)
    top_k_count: int = 0
    duration_ms: float = 0.0


@dataclass
class EvidenceGateResult:
    passed: bool = True
    top_calibrated_confidence: float = 0.0
    top_raw_score: float = 0.0
    threshold: float = 0.80
    retained_evidence: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class GenerationResult:
    answer: str = ""
    grounded: bool = True
    grounding_status: str = "GROUNDED"
    duration_ms: float = 0.0


@dataclass
class VoiceLatencyBreakdown:
    stt_ms: float = 0.0
    guardrail_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranker_ms: float = 0.0
    evidence_gate_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RAGPipelineTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = 0.0
    query: str = ""
    language: str = "unknown"
    stt: SpeechToTextResult = field(default_factory=SpeechToTextResult)
    guardrail: GuardrailResult = field(default_factory=GuardrailResult)
    retrieval: DenseRetrievalResult = field(default_factory=DenseRetrievalResult)
    reranking: RerankingResult = field(default_factory=RerankingResult)
    evidence_gate: EvidenceGateResult = field(default_factory=EvidenceGateResult)
    generation: GenerationResult = field(default_factory=GenerationResult)
    latency: VoiceLatencyBreakdown = field(default_factory=VoiceLatencyBreakdown)
    final_status: str = "COMPLETED"


@dataclass
class PipelineResponse:
    success: bool
    query: str
    answer: str
    sources: List[RetrievedSource] = field(default_factory=list)
    error: Optional[str] = None
    grounded: bool = True
    trace_id: str = ""
    latency: VoiceLatencyBreakdown = field(default_factory=VoiceLatencyBreakdown)


@dataclass
class VoicePipelineResponse:
    success: bool
    transcript: str
    language_code: str
    answer: str
    grounded: bool
    sources: List[RetrievedSource] = field(default_factory=list)
    latency: VoiceLatencyBreakdown = field(default_factory=VoiceLatencyBreakdown)
    trace_id: str = ""
    error: Optional[str] = None
    trace: Optional[RAGPipelineTrace] = None