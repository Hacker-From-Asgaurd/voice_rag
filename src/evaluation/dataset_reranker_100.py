import sys
import time
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


DATASET = Path("data/hindi_dev.parquet")
N = 100
CANDIDATES = 10


def main():

    print("=" * 60)
    print("E5 + RERANKER - 100 QUERY BENCHMARK")
    print("=" * 60)

    df = pd.read_parquet(
        DATASET,
        columns=["query_id", "query", "Answer", "passages"]
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

    sample = valid.head(N)

    print("Dataset rows    :", len(df))
    print("Answerable      :", len(answerable))
    print("Valid benchmark :", len(valid))
    print("Testing         :", len(sample))

    print("\nLoading E5...")
    retriever = Retriever()

    print("Loading reranker...")
    reranker = Reranker()

    recall_1 = 0
    recall_5 = 0
    latencies = []

    for _, row in sample.iterrows():

        query = row["query"]
        passages = row["passages"]

        selected_ids = [
            i
            for i, flag in enumerate(
                passages["is_selected"]
            )
            if int(flag) == 1
        ]

        start = time.perf_counter()

        candidates = retriever.search(
            query,
            top_k=CANDIDATES
        )

        results = reranker.rerank(
            query,
            candidates,
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

        if (
            retrieved_ids
            and retrieved_ids[0] in selected_ids
        ):
            recall_1 += 1

        if any(
            pid in selected_ids
            for pid in retrieved_ids[:5]
        ):
            recall_5 += 1

    latencies.sort()

    total = len(sample)

    avg = sum(latencies) / total
    median = latencies[total // 2]

    p95_index = min(
        int(total * 0.95),
        total - 1
    )

    p95 = latencies[p95_index]

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print("Queries       :", total)

    print(
        f"Recall@1      : "
        f"{recall_1 / total * 100:.2f}%"
    )

    print(
        f"Recall@5      : "
        f"{recall_5 / total * 100:.2f}%"
    )

    print(
        f"Avg latency   : "
        f"{avg:.2f} ms"
    )

    print(
        f"Median latency: "
        f"{median:.2f} ms"
    )

    print(
        f"P95 latency   : "
        f"{p95:.2f} ms"
    )

    print(
        f"Max latency   : "
        f"{max(latencies):.2f} ms"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()