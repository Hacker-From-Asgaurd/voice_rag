from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QueryRequest:
    query: str
    top_k: int = 5


@dataclass
class RetrievedSource:
    score: float
    chunk: str
    query_id: int
    passage_id: int
    is_selected: int


@dataclass
class PipelineResponse:
    success: bool
    query: str
    answer: str
    sources: List[RetrievedSource] = field(default_factory=list)
    error: Optional[str] = None
    grounded: bool = True


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
class VoicePipelineResponse:
    success: bool
    transcript: str
    language_code: str
    answer: str
    grounded: bool
    sources: List[RetrievedSource] = field(default_factory=list)
    latency: VoiceLatencyBreakdown = field(default_factory=VoiceLatencyBreakdown)
    error: Optional[str] = None