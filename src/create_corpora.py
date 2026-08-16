import json
import duckdb

from chunking import (
    fixed_size_chunks,
    sentence_aware_chunks,
    adaptive_parent_child_chunks,
)

INPUT_FILE = "data/hindi_dev.parquet"

OUTPUTS = {
    "fixed": "data/chunks_fixed.jsonl",
    "sentence": "data/chunks_sentence.jsonl",
    "adaptive": "data/chunks_adaptive.jsonl",
}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


print("Loading local Hindi dataset...")

con = duckdb.connect()

rows = con.execute(f"""
    SELECT
        query_id,
        query,
        passages
    FROM '{INPUT_FILE}'
""").fetchall()

print(f"Rows loaded: {len(rows)}")

fixed_rows = []
sentence_rows = []
adaptive_rows = []

chunk_id = 0

for row_number, (query_id, query, passages) in enumerate(rows, start=1):

    translated_passages = passages["Translated_passages"]
    selected_labels = passages["is_selected"]

    for passage_id, (passage, selected) in enumerate(
        zip(translated_passages, selected_labels)
    ):

        if not passage or not passage.strip():
            continue

        passage = passage.strip()

        # -------------------------------------------------
        # Strategy A: Fixed-size
        # -------------------------------------------------

        chunks = fixed_size_chunks(passage)

        for chunk_number, chunk in enumerate(chunks):

            fixed_rows.append({
                "chunk_id": f"fixed_{chunk_id}",
                "query_id": query_id,
                "query": query,
                "passage_id": passage_id,
                "chunk_number": chunk_number,
                "is_selected": selected,
                "chunk": chunk,
                "parent": passage,
            })

            chunk_id += 1

        # -------------------------------------------------
        # Strategy B: Sentence-aware
        # -------------------------------------------------

        chunks = sentence_aware_chunks(passage)

        for chunk_number, chunk in enumerate(chunks):

            sentence_rows.append({
                "query_id": query_id,
                "query": query,
                "passage_id": passage_id,
                "chunk_number": chunk_number,
                "is_selected": selected,
                "chunk": chunk,
                "parent": passage,
            })

        # -------------------------------------------------
        # Strategy C: Adaptive parent-child
        # -------------------------------------------------

        chunks = adaptive_parent_child_chunks(passage)

        for chunk_number, item in enumerate(chunks):

            adaptive_rows.append({
                "query_id": query_id,
                "query": query,
                "passage_id": passage_id,
                "chunk_number": chunk_number,
                "is_selected": selected,
                "chunk": item["chunk"],
                "parent": item["parent"],
                "is_child": item["is_child"],
            })

    if row_number % 500 == 0:
        print(f"Processed {row_number}/{len(rows)} rows...")


print("\nWriting corpora...")

write_jsonl(OUTPUTS["fixed"], fixed_rows)
write_jsonl(OUTPUTS["sentence"], sentence_rows)
write_jsonl(OUTPUTS["adaptive"], adaptive_rows)

print("\nDone.")

print("Fixed chunks:", len(fixed_rows))
print("Sentence chunks:", len(sentence_rows))
print("Adaptive chunks:", len(adaptive_rows))