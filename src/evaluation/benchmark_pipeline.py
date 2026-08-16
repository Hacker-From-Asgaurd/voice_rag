import sys
import time
import statistics
from pathlib import Path

# Add src/ to Python path
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.generator import generate_answer


# ---------------------------------------------------------
# Test queries
# ---------------------------------------------------------

QUERIES = [
    "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?",
    "भारत में क्रिकेट के सबसे प्रसिद्ध खिलाड़ी कौन हैं?",
    "What is artificial intelligence?",
    "मैनहट्टन परियोजना क्या थी?",
    "द्वितीय विश्व युद्ध में मैनहट्टन परियोजना की क्या भूमिका थी?",
    "What was the purpose of the Manhattan Project?",
    "मैनहट्टन परियोजना कब शुरू हुई?",
    "What is machine learning?",
    "परमाणु बम का विकास किस परियोजना के अंतर्गत हुआ?",
    "What are the applications of artificial intelligence?",
]


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FAISS_TOP_K = 10
FINAL_TOP_K = 10

FAISS_THRESHOLD = 0.85

RERANK_THRESHOLDS = [
    -2.0,
    0.0,
    1.0,
    1.5,
    2.0,
]


# ---------------------------------------------------------
# Percentile
# ---------------------------------------------------------

def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1
    )

    weight = index - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        ) * weight
    )


# ---------------------------------------------------------
# Build context
# ---------------------------------------------------------

def build_context(results):

    parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        parts.append(
            f"[Source {i}]\n"
            f"{result['chunk']}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("END-TO-END RAG THRESHOLD BENCHMARK")
    print("=" * 70)

    print("\nLoading retriever...")

    retriever = Retriever()

    print("\nLoading reranker...")

    reranker = Reranker()

    print("\nModels ready.")

    print(
        f"\nQueries: {len(QUERIES)}"
    )

    print(
        f"FAISS Top-K: {FAISS_TOP_K}"
    )

    print(
        f"Final Top-K: {FINAL_TOP_K}"
    )

    # -----------------------------------------------------
    # Store results for each threshold
    # -----------------------------------------------------

    benchmark = {}

    for threshold in RERANK_THRESHOLDS:

        benchmark[threshold] = {
            "latencies": [],
            "retrieval_times": [],
            "rerank_times": [],
            "generation_times": [],
            "answered": 0,
            "abstained": 0,
            "empty": 0,
        }

    # -----------------------------------------------------
    # Evaluate every query
    # -----------------------------------------------------

    for query_number, query in enumerate(
        QUERIES,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"[{query_number}/{len(QUERIES)}] "
            f"{query}"
        )

        # =================================================
        # RETRIEVAL
        # =================================================

        retrieval_start = time.perf_counter()

        candidates = retriever.search(
            query,
            top_k=FAISS_TOP_K
        )

        retrieval_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        # =================================================
        # RERANK
        # =================================================

        rerank_start = time.perf_counter()

        reranked = reranker.rerank(
            query,
            candidates,
            top_k=FINAL_TOP_K
        )

        rerank_ms = (
            time.perf_counter()
            - rerank_start
        ) * 1000

        print(
            f"Candidates : {len(candidates)}"
        )

        print(
            f"Reranked   : {len(reranked)}"
        )

        print(
            f"Retrieval  : {retrieval_ms:.2f} ms"
        )

        print(
            f"Reranking  : {rerank_ms:.2f} ms"
        )

        # =================================================
        # TEST EACH RERANK THRESHOLD
        # =================================================

        for threshold in RERANK_THRESHOLDS:

            config = benchmark[
                threshold
            ]

            # -------------------------------------------------
            # Evidence gate
            #
            # IMPORTANT:
            # FAISS threshold remains 0.85.
            # Rerank threshold is an additional gate.
            # -------------------------------------------------

            evidence = [
                result
                for result in reranked
                if (
                    float(result["score"])
                    >= FAISS_THRESHOLD
                    and
                    float(result["rerank_score"])
                    >= threshold
                )
            ]

            # -------------------------------------------------
            # Abstention
            # -------------------------------------------------

            if not evidence:

                config["abstained"] += 1

                config["latencies"].append(
                    retrieval_ms + rerank_ms
                )

                config["retrieval_times"].append(
                    retrieval_ms
                )

                config["rerank_times"].append(
                    rerank_ms
                )

                config["generation_times"].append(
                    0.0
                )

                continue

            # -------------------------------------------------
            # Build context
            # -------------------------------------------------

            context = build_context(
                evidence
            )

            # -------------------------------------------------
            # Generation
            # -------------------------------------------------

            generation_start = (
                time.perf_counter()
            )

            try:

                answer = generate_answer(
                    query,
                    context
                )

            except Exception as e:

                print(
                    f"[Generation Error] {e}"
                )

                answer = ""

            generation_ms = (
                time.perf_counter()
                - generation_start
            ) * 1000

            total_ms = (
                retrieval_ms
                + rerank_ms
                + generation_ms
            )

            # -------------------------------------------------
            # Validate answer
            # -------------------------------------------------

            if not answer or not answer.strip():

                config["empty"] += 1

            else:

                config["answered"] += 1

            config["latencies"].append(
                total_ms
            )

            config["retrieval_times"].append(
                retrieval_ms
            )

            config["rerank_times"].append(
                rerank_ms
            )

            config["generation_times"].append(
                generation_ms
            )

    # ======================================================
    # RESULTS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("END-TO-END RESULTS")
    print("=" * 70)

    for threshold in RERANK_THRESHOLDS:

        config = benchmark[
            threshold
        ]

        total_queries = len(
            QUERIES
        )

        answered = config[
            "answered"
        ]

        abstained = config[
            "abstained"
        ]

        empty = config[
            "empty"
        ]

        print("\n" + "-" * 70)

        print(
            f"RERANK THRESHOLD: "
            f"{threshold:.2f}"
        )

        print("-" * 70)

        print(
            f"Answered          : "
            f"{answered}/{total_queries}"
        )

        print(
            f"Abstained         : "
            f"{abstained}/{total_queries}"
        )

        print(
            f"Empty generations : "
            f"{empty}/{total_queries}"
        )

        print(
            f"Answer rate       : "
            f"{answered / total_queries:.4f}"
        )

        print(
            f"Abstention rate   : "
            f"{abstained / total_queries:.4f}"
        )

        # -------------------------------------------------
        # Latency
        # -------------------------------------------------

        print("\nLatency")

        print(
            f"Average : "
            f"{statistics.mean(config['latencies']):.2f} ms"
        )

        print(
            f"P50     : "
            f"{percentile(config['latencies'], 0.50):.2f} ms"
        )

        print(
            f"P95     : "
            f"{percentile(config['latencies'], 0.95):.2f} ms"
        )

        print(
            f"P100    : "
            f"{max(config['latencies']):.2f} ms"
        )

        print("\nComponent latency")

        print(
            f"Retrieval : "
            f"{statistics.mean(config['retrieval_times']):.2f} ms"
        )

        print(
            f"Reranking : "
            f"{statistics.mean(config['rerank_times']):.2f} ms"
        )

        print(
            f"Generation: "
            f"{statistics.mean(config['generation_times']):.2f} ms"
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        "Threshold | Answered | Abstain | "
        "Answer Rate | Avg Latency"
    )

    print("-" * 70)

    for threshold in RERANK_THRESHOLDS:

        config = benchmark[
            threshold
        ]

        total_queries = len(
            QUERIES
        )

        answered = config[
            "answered"
        ]

        abstained = config[
            "abstained"
        ]

        answer_rate = (
            answered
            / total_queries
        )

        avg_latency = statistics.mean(
            config["latencies"]
        )

        print(
            f"{threshold:8.2f} | "
            f"{answered:8d} | "
            f"{abstained:7d} | "
            f"{answer_rate:11.3f} | "
            f"{avg_latency:11.2f} ms"
        )

    print("=" * 70)

    print("\nIMPORTANT:")
    print(
        "This benchmark measures actual end-to-end "
        "retrieval + reranking + generation behavior."
    )

    print(
        "It does NOT automatically determine answer "
        "correctness. Human/ground-truth evaluation "
        "is still required for answer quality."
    )


if __name__ == "__main__":
    main()