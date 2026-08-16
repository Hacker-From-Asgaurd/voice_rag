from typing import List, Optional
from pydantic import BaseModel, Field


class TextQueryRequest(BaseModel):
    query: str = Field(..., description="User query text")


class RetrievedSourceSchema(BaseModel):
    score: float
    chunk: str
    query_id: int
    passage_id: int
    is_selected: int


class TextLatencySchema(BaseModel):
    guardrail_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranker_ms: float = 0.0
    evidence_gate_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class TextQueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    grounded: bool
    sources: List[RetrievedSourceSchema] = []
    latency: TextLatencySchema
    error: Optional[str] = None


class VoiceLatencySchema(BaseModel):
    stt_ms: float = 0.0
    guardrail_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranker_ms: float = 0.0
    evidence_gate_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class VoiceQueryResponse(BaseModel):
    success: bool
    transcript: str
    language_code: str
    answer: str
    grounded: bool
    sources: List[RetrievedSourceSchema] = []
    latency: VoiceLatencySchema
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    rag: bool
    sarvam: bool
    generator: bool


class MetricsResponse(BaseModel):
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_latency_ms: float
    p50_ms: float
    p70_ms: float
    p95_ms: float
    p100_ms: float
    candidate_k: int
    rerank_top_k: int
    evidence_threshold: float
    total_queries_benchmarked: int
