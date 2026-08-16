import sys
import time
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from harness.pipeline import RAGPipeline
from harness.schemas import QueryRequest


DATASET = Path("data/hindi_dev.parquet")


def main():

    print("=" * 70)
    print("DATASET RETRIEVAL BASELINE")
    print("=" * 70)

    df = pd.read_parquet(
        DATASET,
        columns=[
            "query_id",
            "query",
            "Answer",
            "passages"
        ]
    )

    # --------------------------------------------------
    # Answerable rows with valid selected passage
    # --------------------------------------------------

    answerable = df[
        ~df["Answer"].astype(str).str.contains(
            "No Answer Present|कोई उत्तर नहीं मिला",
            case=False,
            regex=True
        )
    ]

    valid = answerable[
        answerable["passages"].apply(
            lambda x: 1 in list(x["is_selected"])
        )
    ]

    # Fixed deterministic sample
    sample = valid.head(10)

    print(f"Dataset rows      : {len(df)}")
    print(f"Answerable rows   : {len(answerable)}")
    print(f"Valid benchmark   : {len(valid)}")
    print(f"Testing           : {len(sample)}")

    print("\nInitializing pipeline...")
    pipeline = RAGPipeline()

    recall_1 = 0
    recall_5 = 0
    total = 0

    latencies = []

    print("\n" + "=" * 70)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):

        query = row["query"]
        passages = row["passages"]

        selected_ids = [
            idx
            for idx, flag in enumerate(
                passages["is_selected"]
            )
            if int(flag) == 1
        ]

        print("\n" + "-" * 70)
        print(f"TEST {i}/10")
        print(f"QUERY ID : {row['query_id']}")
        print(f"QUERY    : {query}")
        print(f"GROUND TRUTH PASSAGES : {selected_ids}")

        start = time.perf_counter()

        results = pipeline.retrieve(
            QueryRequest(
                query=query,
                top_k=5
            )
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

        retrieved_ids = [
            int(x["passage_id"])
            for x in results
        ]

        print(f"RETRIEVED PASSAGES      : {retrieved_ids}")

        print("\nRERANK SCORES:")

        for x in results:
            print(
                f"  passage={x['passage_id']} "
                f"score={float(x['rerank_score']):.4f}"
            )

        hit_1 = (
            len(retrieved_ids) >= 1
            and retrieved_ids[0] in selected_ids
        )

        hit_5 = any(
            pid in selected_ids
            for pid in retrieved_ids[:5]
        )

        if hit_1:
            recall_1 += 1

        if hit_5:
            recall_5 += 1

        total += 1

        print(f"\nRecall@1 : {'HIT' if hit_1 else 'MISS'}")
        print(f"Recall@5 : {'HIT' if hit_5 else 'MISS'}")
        print(f"Latency  : {elapsed:.2f} ms")

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET BASELINE RESULTS")
    print("=" * 70)

    print(f"Tests       : {total}")
    print(
        f"Recall@1    : "
        f"{recall_1 / total * 100:.2f}%"
    )
    print(
        f"Recall@5    : "
        f"{recall_5 / total * 100:.2f}%"
    )

    if latencies:
        print(
            f"Avg latency : "
            f"{sum(latencies) / len(latencies):.2f} ms"
        )
        print(
            f"Max latency : "
            f"{max(latencies):.2f} ms"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()