import regex
def fixed_size_chunks(text, chunk_size=1000, overlap=150):
    """
    Split text into fixed-size character chunks with overlap.
    """

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks



def sentence_aware_chunks(
    text,
    target_size=1000,
    max_size=1400
):
    """
    Split text into chunks while preserving sentence boundaries.
    """

    if not text or not text.strip():
        return []

    # Split after common sentence-ending punctuation.
    sentences = regex.split(
        r"(?<=[.!?।])\s+",
        text.strip()
    )

    chunks = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # If adding this sentence stays within our preferred size,
        # keep it in the current chunk.
        if not current:
            current = sentence

        elif len(current) + 1 + len(sentence) <= target_size:
            current += " " + sentence

        # If the current chunk is already reasonably sized,
        # start a new chunk.
        elif len(current) <= max_size:
            chunks.append(current)
            current = sentence

        # Extremely long individual sentence.
        else:
            chunks.append(current)

            if len(sentence) <= max_size:
                current = sentence
            else:
                # Fall back to fixed-size splitting for
                # an unusually long sentence.
                long_chunks = fixed_size_chunks(
                    sentence,
                    chunk_size=target_size,
                    overlap=100
                )

                chunks.extend(long_chunks[:-1])
                current = long_chunks[-1]

    if current:
        chunks.append(current)

    return chunks

def adaptive_parent_child_chunks(
    text,
    parent_threshold=1200,
    target_size=1000,
    max_size=1400
):
    """
    Keep short passages intact.
    Split long passages using sentence-aware chunking.

    Returns:
        A list of dictionaries containing:
        - chunk: text used for retrieval
        - parent: original passage
        - is_child: whether the passage was split
    """

    if not text or not text.strip():
        return []

    text = text.strip()

    # Short passage: keep the complete passage.
    if len(text) <= parent_threshold:
        return [
            {
                "chunk": text,
                "parent": text,
                "is_child": False
            }
        ]

    # Long passage: split into sentence-aware child chunks.
    children = sentence_aware_chunks(
        text,
        target_size=target_size,
        max_size=max_size
    )

    return [
        {
            "chunk": child,
            "parent": text,
            "is_child": True
        }
        for child in children
    ]