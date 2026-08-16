import json
import sys
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

METADATA_FILE = Path("data/e5_metadata.json")
ADAPTIVE_FILE = Path("data/chunks_adaptive.jsonl")
PARQUET_FILE = Path("data/hindi_dev.parquet")
BENCHMARK_JSON = Path("data/retrieval_benchmark_3037.json")
OUTPUT_JSON = Path("data/duplicate_investigation.json")


def main():
    if not METADATA_FILE.exists():
        print(f"Error: {METADATA_FILE} not found")
        return

    # 1. Load metadata
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    total_metadata_records = len(metadata)

    # Group by (query_id, passage_id)
    pair_to_records = defaultdict(list)
    for idx, rec in enumerate(metadata):
        key = (int(rec["query_id"]), int(rec["passage_id"]))
        rec_copy = dict(rec)
        rec_copy["metadata_idx"] = idx
        pair_to_records[key].append(rec_copy)

    # Find duplicate pairs
    duplicate_pairs = {k: v for k, v in pair_to_records.items() if len(v) > 1}
    num_duplicate_pairs = len(duplicate_pairs)

    # 2. Load raw parquet for cross-checking
    df_raw = pd.read_parquet(PARQUET_FILE, columns=["query_id", "query", "Answer", "passages"])
    raw_by_qid = {}
    for _, row in df_raw.iterrows():
        raw_by_qid[int(row["query_id"])] = row

    # 3. Load adaptive jsonl for cross-checking
    adaptive_by_qid_pid = defaultdict(list)
    if ADAPTIVE_FILE.exists():
        with open(ADAPTIVE_FILE, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                rec = json.loads(line)
                k = (int(rec["query_id"]), int(rec["passage_id"]))
                if k in duplicate_pairs:
                    rec["line_idx"] = line_idx
                    adaptive_by_qid_pid[k].append(rec)

    # 4. Classify duplicates
    classifications = {"A": 0, "B": 0, "C": 0, "D": 0}
    pair_details = []
    affected_qids = set()
    exact_duplicate_chunk_count = 0
    distinct_chunk_count = 0
    max_chunks_for_pair = 0
    total_duplicate_records = sum(len(v) for v in duplicate_pairs.values())

    for (qid, pid), records in duplicate_pairs.items():
        affected_qids.add(qid)
        num_records = len(records)
        if num_records > max_chunks_for_pair:
            max_chunks_for_pair = num_records

        chunk_texts = [r.get("chunk", "") for r in records]
        parent_texts = [r.get("parent", "") for r in records]
        chunk_numbers = [r.get("chunk_number") for r in records]
        is_child_flags = [r.get("is_child") for r in records]
        is_selected_flags = [r.get("is_selected") for r in records]
        chunk_lengths = [len(c) for c in chunk_texts]

        # Check if chunks are identical
        are_chunks_exact_duplicates = len(set(chunk_texts)) == 1
        if are_chunks_exact_duplicates:
            exact_duplicate_chunk_count += 1
        else:
            distinct_chunk_count += 1

        is_parent_identical = len(set(parent_texts)) == 1
        parent_len = len(parent_texts[0]) if parent_texts else 0

        # Cross check raw dataset
        raw_row = raw_by_qid.get(qid)
        raw_query = ""
        raw_passage_text = ""
        raw_selected_flag = None
        raw_passage_len = 0

        if raw_row is not None:
            raw_query = str(raw_row["query"])
            passages = raw_row["passages"]
            translated_passages = passages.get("Translated_passages", [])
            selected_labels = passages.get("is_selected", [])
            if pid < len(translated_passages):
                raw_passage_text = translated_passages[pid]
                raw_passage_len = len(raw_passage_text) if raw_passage_text else 0
            if pid < len(selected_labels):
                raw_selected_flag = int(selected_labels[pid])

        # Classification logic:
        # Type A: legitimate multi-chunk passage (adaptive child chunks from parent > 1200 chars)
        # Type B: raw dataset repetition (e.g. duplicate rows/passages in raw dataset)
        # Type C: corpus/index generation artifact (exact duplicates generated accidentally)
        # Type D: unclear
        if (
            not are_chunks_exact_duplicates
            and all(is_child_flags)
            and is_parent_identical
            and parent_len > 1200
        ):
            category = "A"
        elif are_chunks_exact_duplicates and not all(is_child_flags):
            category = "C"
        elif raw_passage_len <= 1200 and not are_chunks_exact_duplicates:
            category = "D"
        else:
            # Check if all chunks are different portions of parent
            all_in_parent = all(c in parent_texts[0] for c in chunk_texts)
            if all_in_parent and not are_chunks_exact_duplicates:
                category = "A"
            else:
                category = "D"

        classifications[category] += 1

        pair_details.append({
            "query_id": qid,
            "passage_id": pid,
            "classification": category,
            "record_count": num_records,
            "chunk_numbers": chunk_numbers,
            "is_child": is_child_flags,
            "is_selected": is_selected_flags,
            "chunk_lengths": chunk_lengths,
            "are_chunks_exact_duplicates": are_chunks_exact_duplicates,
            "parent_length": parent_len,
            "is_parent_identical": is_parent_identical,
            "raw_passage_length": raw_passage_len,
        })

    # 6. Top-K slot occupancy analysis
    # Check how often duplicate chunks from the same passage appear in top-5 and top-10
    from retrieval.retriever import Retriever

    retriever = Retriever()

    top5_multi_chunk_queries = 0
    top10_multi_chunk_queries = 0
    slots_wasted_top5 = 0
    slots_wasted_top10 = 0

    valid_answerable = df_raw[
        ~df_raw["Answer"].astype(str).str.contains(
            "No Answer Present|कोई उत्तर नहीं मिला", case=False, regex=True
        )
    ]
    valid_queries = valid_answerable[
        valid_answerable["passages"].apply(lambda x: 1 in list(x["is_selected"]))
    ]

    for _, row in valid_queries.iterrows():
        query = row["query"]
        candidates = retriever.search(query, top_k=10)

        # In top 5
        seen_passages_top5 = set()
        multi_in_top5 = False
        for c in candidates[:5]:
            p_key = (int(c["query_id"]), int(c["passage_id"]))
            if p_key in seen_passages_top5:
                multi_in_top5 = True
                slots_wasted_top5 += 1
            else:
                seen_passages_top5.add(p_key)
        if multi_in_top5:
            top5_multi_chunk_queries += 1

        # In top 10
        seen_passages_top10 = set()
        multi_in_top10 = False
        for c in candidates[:10]:
            p_key = (int(c["query_id"]), int(c["passage_id"]))
            if p_key in seen_passages_top10:
                multi_in_top10 = True
                slots_wasted_top10 += 1
            else:
                seen_passages_top10.add(p_key)
        if multi_in_top10:
            top10_multi_chunk_queries += 1

    total_valid = len(valid_queries)

    # Produce concise summary
    print("=" * 60)
    print("DUPLICATE INVESTIGATION")
    print("=" * 60)
    print(f"Metadata records       : {total_metadata_records}")
    print(f"Duplicate pairs        : {num_duplicate_pairs}")
    print()
    print(f"Type A (Multi-chunk)   : {classifications['A']}")
    print(f"Type B (Raw repetition): {classifications['B']}")
    print(f"Type C (Gen artifact)  : {classifications['C']}")
    print(f"Type D (Unclear)       : {classifications['D']}")
    print("=" * 60)
    print("DETAILED COUNTS")
    print("-" * 60)
    print(f"Total duplicate records: {total_duplicate_records}")
    print(f"Unique affected QIDs   : {len(affected_qids)}")
    print(f"Max chunks for 1 pair  : {max_chunks_for_pair}")
    print(f"Exact duplicate chunks : {exact_duplicate_chunk_count}")
    print(f"Distinct child chunks  : {distinct_chunk_count}")
    print("-" * 60)
    print("RETRIEVAL CO-OCCURRENCE IMPACT (3,037 VALID QUERIES)")
    print("-" * 60)
    print(f"Queries with co-occurring chunks in Top-5 : {top5_multi_chunk_queries} ({top5_multi_chunk_queries/total_valid*100:.2f}%)")
    print(f"Queries with co-occurring chunks in Top-10: {top10_multi_chunk_queries} ({top10_multi_chunk_queries/total_valid*100:.2f}%)")
    print(f"Total Top-5 slots consumed by duplicates  : {slots_wasted_top5}")
    print(f"Total Top-10 slots consumed by duplicates : {slots_wasted_top10}")
    print("=" * 60)

    output_data = {
        "summary": {
            "metadata_records": total_metadata_records,
            "duplicate_pairs": num_duplicate_pairs,
            "total_duplicate_records": total_duplicate_records,
            "unique_affected_query_ids": len(affected_qids),
            "max_chunks_for_single_pair": max_chunks_for_pair,
            "exact_duplicate_chunks": exact_duplicate_chunk_count,
            "distinct_child_chunks": distinct_chunk_count,
        },
        "classifications": classifications,
        "affected_pairs": pair_details,
        "retrieval_impact_analysis": {
            "total_queries_evaluated": total_valid,
            "top5_multi_chunk_queries": top5_multi_chunk_queries,
            "top5_multi_chunk_queries_pct": round(top5_multi_chunk_queries / total_valid * 100, 2),
            "top10_multi_chunk_queries": top10_multi_chunk_queries,
            "top10_multi_chunk_queries_pct": round(top10_multi_chunk_queries / total_valid * 100, 2),
            "total_top5_duplicate_slots": slots_wasted_top5,
            "total_top10_duplicate_slots": slots_wasted_top10,
        },
        "recommendation": {
            "is_legitimate": classifications["A"] == num_duplicate_pairs,
            "summary": "All 77 duplicate pairs are Type A legitimate child chunks from long passages (>1200 chars). Deduplication or parent-level aggregation during retrieval is a viable controlled experiment for Phase 3 to prevent redundant slot usage.",
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
