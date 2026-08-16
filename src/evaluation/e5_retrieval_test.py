import sys
import time
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.retriever import Retriever


DATASET = Path("data/hindi_dev.parquet")


def main():

    print("=" * 60)
    print("E5-ONLY RETRIEVAL BENCHMARK")
    print("=" * 60)

    df = pd.read_parquet(
        DATASET,
        columns=[
            "query_id",
            "query",
            "Answer",
            "passages"
        ]
    )

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

    sample = valid.head(10)

    print("Dataset rows    :", len(df))
    print("Answerable      :", len(answerable))
    print("Valid benchmark :", len(valid))
    print("Testing         :", len(sample))

    print("\nLoading E5 retriever only...")

    retriever = Retriever()

    recall_1 = 0
    recall_5 = 0
    latencies = []

    for i, (_, row) in enumerate(
        sample.iterrows(),
        start=1
    ):

        query = row["query"]
        passages = row["passages"]

        selected_ids = [
            idx
            for idx, flag in enumerate(
                passages["is_selected"]
            )
            if int(flag) == 1
        ]

        start = time.perf_counter()

        results = retriever.search(
            query,
            top_k=5
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

        retrieved_ids = [
            int(x["passage_id"])
            for x in results
        ]

        hit_1 = (
            len(retrieved_ids) > 0
            and retrieved_ids[0] in selected_ids
        )

        hit_5 = any(
            pid in selected_ids
            for pid in retrieved_ids
        )

        if hit_1:
            recall_1 += 1

        if hit_5:
            recall_5 += 1

        print(
            f"TEST {i:02d} | "
            f"GT={selected_ids} | "
            f"RET={retrieved_ids} | "
            f"R@1={'HIT' if hit_1 else 'MISS'} | "
            f"R@5={'HIT' if hit_5 else 'MISS'} | "
            f"{elapsed:.1f} ms"
        )

    total = len(sample)

    print("\n" + "=" * 60)
    print("E5-ONLY RESULTS")
    print("=" * 60)

    print(
        f"Recall@1    : "
        f"{recall_1 / total * 100:.2f}%"
    )

    print(
        f"Recall@5    : "
        f"{recall_5 / total * 100:.2f}%"
    )

    print(
        f"Avg latency : "
        f"{sum(latencies) / len(latencies):.2f} ms"
    )

    print(
        f"Max latency : "
        f"{max(latencies):.2f} ms"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()