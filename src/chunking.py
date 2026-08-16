import re
from typing import List, Dict, Any, Optional
import numpy as np
import regex

# ---------------------------------------------------------------------------
# 1. FIXED-SIZE CHUNKING (Baseline)
# ---------------------------------------------------------------------------

def fixed_size_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Split text into fixed-size character chunks with controlled overlap.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# 2. SENTENCE-AWARE CHUNKING
# ---------------------------------------------------------------------------

def sentence_aware_chunks(
    text: str,
    target_size: int = 1000,
    max_size: int = 1400
) -> List[str]:
    """
    Split text into chunks while preserving sentence boundaries across
    multiple Indic and Latin scripts (including Purna Viram '।').
    """
    if not text or not text.strip():
        return []

    # Split after sentence-ending punctuation: period, exclamation, question mark, devanagari danda
    sentences = regex.split(r"(?<=[.!?।])\s+", text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= target_size:
            current += " " + sentence
        elif len(current) <= max_size:
            chunks.append(current)
            current = sentence
        else:
            chunks.append(current)
            if len(sentence) <= max_size:
                current = sentence
            else:
                long_chunks = fixed_size_chunks(sentence, chunk_size=target_size, overlap=100)
                chunks.extend(long_chunks[:-1])
                current = long_chunks[-1]

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# 3. SEMANTIC BOUNDARY CHUNKING (Embedding-Assisted Distance Detection)
# ---------------------------------------------------------------------------

def semantic_boundary_chunks(
    text: str,
    embedder=None,
    distance_threshold: float = 0.35,
    target_size: int = 1000,
    max_size: int = 1400
) -> List[str]:
    """
    Splits text by identifying semantic shifts between adjacent sentences
    using embedding cosine distance. If no embedder is provided, falls back
    to sentence-aware boundaries.
    """
    if not text or not text.strip():
        return []

    sentences = [s.strip() for s in regex.split(r"(?<=[.!?।])\s+", text.strip()) if s.strip()]
    if len(sentences) <= 1 or embedder is None:
        return sentence_aware_chunks(text, target_size=target_size, max_size=max_size)

    try:
        # Encode sentences to compute adjacent cosine similarities
        embeddings = embedder.encode(sentences, normalize_embeddings=True)
        # Cosine distance between adjacent sentence pairs
        adjacent_dists = [
            1.0 - float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_len = len(sentences[0])

        for i, dist in enumerate(adjacent_dists):
            next_sentence = sentences[i + 1]
            next_len = len(next_sentence)

            is_semantic_shift = dist > distance_threshold
            exceeds_target = (current_len + next_len) > target_size

            if (is_semantic_shift and current_len >= 200) or (exceeds_target and current_len >= 400):
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [next_sentence]
                current_len = next_len
            else:
                current_chunk_sentences.append(next_sentence)
                current_len += 1 + next_len

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    except Exception:
        return sentence_aware_chunks(text, target_size=target_size, max_size=max_size)


# ---------------------------------------------------------------------------
# 4. PARENT-CHILD CHUNK STORE (Metadata-Aware Reference Layer)
# ---------------------------------------------------------------------------

class ParentChildChunkStore:
    """
    Memory-efficient metadata-aware chunk registry.
    Stores lightweight vector metadata referencing parent_passage_id
    without duplicating raw parent passage text across vector records.
    """

    def __init__(self):
        self.parent_passages: Dict[int, str] = {}
        self.chunk_metadata: List[Dict[str, Any]] = []

    def register_passage(
        self,
        passage_id: int,
        passage_text: str,
        query_id: int = 0,
        language: str = "hi",
        strategy: str = "sentence_aware",
        embedder=None
    ) -> List[Dict[str, Any]]:
        """
        Registers a parent passage and produces metadata-rich chunk descriptors.
        """
        self.parent_passages[passage_id] = passage_text

        if strategy == "fixed_size":
            chunks = fixed_size_chunks(passage_text)
        elif strategy == "semantic_boundary":
            chunks = semantic_boundary_chunks(passage_text, embedder=embedder)
        else:
            chunks = sentence_aware_chunks(passage_text)

        created = []
        for idx, chunk_text in enumerate(chunks):
            record = {
                "chunk_id": len(self.chunk_metadata),
                "passage_id": passage_id,
                "parent_passage_id": passage_id,
                "query_id": query_id,
                "language": language,
                "chunk_position": idx,
                "chunk_text": chunk_text,
                "is_child": len(chunks) > 1,
            }
            self.chunk_metadata.append(record)
            created.append(record)

        return created

    def get_parent(self, parent_passage_id: int) -> Optional[str]:
        return self.parent_passages.get(parent_passage_id)