import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
# Add src/ to Python path
# ---------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_FILE = "data/hindi_dev.parquet"

NUM_QUERIES = 50

# Validated configuration from previous experiments:
# E5 FAISS Top-10 -> CrossEncoder -> Top-10
INITIAL_TOP_K = 10
FINAL_TOP_K = 10

# CrossEncoder score thresholds.
#
# IMPORTANT:
# These are NOT FAISS cosine-similarity thresholds.
# They apply to result["rerank_score"].
#
# We deliberately test a broad range because
# CrossEncoder scores are model-specific.
RERANK_THRESHOLDS = [
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("CROSS-ENCODER RERANKER THRESHOLD BENCHMARK")
    print("=" * 70)

    # -----------------------------------------------------
    # Load evaluation dataset
    # -----------------------------------------------------

    print("\nLoading evaluation dataset...")

    df = pd.read_parquet(
        DATASET_FILE
    )

    df = df.head(
        NUM_QUERIES
    )

    print(
        f"Queries evaluated: {len(df)}"
    )

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    ground_truth = {}

    for _, row in df.iterrows():

        query_id = int(
            row["query_id"]
        )

        selected = (
            row["passages"]["is_selected"]
        )

        relevant_passages = {
            passage_id
            for passage_id, value
            in enumerate(selected)
            if int(value) == 1
        }

        ground_truth[
            query_id
        ] = relevant_passages

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    print("\nLoading E5 retriever...")

    retriever = Retriever()

    print("\nLoading CrossEncoder reranker...")

    reranker = Reranker()

    print("\nModels ready.")

    # -----------------------------------------------------
    # Retrieve + rerank ONCE
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("GENERATING RERANKED RESULTS")
    print("-" * 70)

    all_results = {}

    for query_number, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        query_id = int(
            row["query_id"]
        )

        query = row["query"]

        candidates = retriever.search(
            query,
            top_k=INITIAL_TOP_K
        )

        reranked = reranker.rerank(
            query,
            candidates,
            top_k=FINAL_TOP_K
        )

        all_results[
            query_id
        ] = reranked

        if query_number % 10 == 0:

            print(
                f"Processed "
                f"{query_number}/"
                f"{len(df)} queries..."
            )

    # -----------------------------------------------------
    # Evaluate thresholds
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RERANKER SCORE THRESHOLD RESULTS")
    print("=" * 70)

    results_table = []

    for threshold in RERANK_THRESHOLDS:

        queries_with_evidence = 0

        total_evidence = 0

        relevant_evidence = 0

        false_evidence = 0

        relevant_available = 0

        reciprocal_rank_total = 0.0

        # -------------------------------------------------
        # Evaluate every query
        # -------------------------------------------------

        for query_id, retrieved in all_results.items():

            relevant_passages = ground_truth[
                query_id
            ]

            relevant_available += len(
                relevant_passages
            )

            # ---------------------------------------------
            # IMPORTANT:
            # Filter using CrossEncoder score.
            # ---------------------------------------------

            evidence = [
                result
                for result in retrieved
                if float(
                    result["rerank_score"]
                ) >= threshold
            ]

            if evidence:

                queries_with_evidence += 1

            total_evidence += len(
                evidence
            )

            relevance = []

            for result in evidence:

                retrieved_query_id = int(
                    result["query_id"]
                )

                retrieved_passage_id = int(
                    result["passage_id"]
                )

                is_relevant = (
                    retrieved_query_id == query_id
                    and
                    retrieved_passage_id
                    in relevant_passages
                )

                relevance.append(
                    is_relevant
                )

                if is_relevant:

                    relevant_evidence += 1

                else:

                    false_evidence += 1

            # ---------------------------------------------
            # Reciprocal rank
            # ---------------------------------------------

            for rank, is_relevant in enumerate(
                relevance,
                start=1
            ):

                if is_relevant:

                    reciprocal_rank_total += (
                        1.0 / rank
                    )

                    break

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        num_queries = len(
            all_results
        )

        abstention_rate = (
            1.0
            -
            (
                queries_with_evidence
                /
                num_queries
            )
        )

        if total_evidence > 0:

            precision = (
                relevant_evidence
                /
                total_evidence
            )

        else:

            precision = 0.0

        if relevant_available > 0:

            recall = (
                relevant_evidence
                /
                relevant_available
            )

        else:

            recall = 0.0

        if precision + recall > 0:

            f1 = (
                2
                * precision
                * recall
                /
                (
                    precision
                    +
                    recall
                )
            )

        else:

            f1 = 0.0

        mrr = (
            reciprocal_rank_total
            /
            num_queries
        )

        results_table.append({
            "threshold": threshold,
            "queries_with_evidence":
                queries_with_evidence,
            "abstention_rate":
                abstention_rate,
            "total_evidence":
                total_evidence,
            "relevant_evidence":
                relevant_evidence,
            "false_evidence":
                false_evidence,
            "precision":
                precision,
            "recall":
                recall,
            "f1":
                f1,
            "mrr":
                mrr,
        })

        # -------------------------------------------------
        # Print
        # -------------------------------------------------

        print("\n" + "-" * 70)

        print(
            f"RERANK SCORE THRESHOLD: "
            f"{threshold:.2f}"
        )

        print(
            f"Queries with evidence : "
            f"{queries_with_evidence}/"
            f"{num_queries}"
        )

        print(
            f"Abstention rate        : "
            f"{abstention_rate:.4f}"
        )

        print(
            f"Total evidence         : "
            f"{total_evidence}"
        )

        print(
            f"Relevant evidence      : "
            f"{relevant_evidence}"
        )

        print(
            f"False evidence         : "
            f"{false_evidence}"
        )

        print(
            f"Precision              : "
            f"{precision:.4f}"
        )

        print(
            f"Recall                 : "
            f"{recall:.4f}"
        )

        print(
            f"F1                    : "
            f"{f1:.4f}"
        )

        print(
            f"MRR                   : "
            f"{mrr:.4f}"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Evidence':<12}"
        f"{'Abstain':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'MRR':<12}"
    )

    print("-" * 70)

    for row in results_table:

        print(
            f"{row['threshold']:<12.2f}"
            f"{row['queries_with_evidence']:<12}"
            f"{row['abstention_rate']:<12.3f}"
            f"{row['precision']:<12.3f}"
            f"{row['recall']:<12.3f}"
            f"{row['f1']:<12.3f}"
            f"{row['mrr']:<12.3f}"
        )

    # -----------------------------------------------------
    # Best F1
    # -----------------------------------------------------

    best_f1 = max(
        results_table,
        key=lambda x: x["f1"]
    )

    # -----------------------------------------------------
    # Best MRR
    # -----------------------------------------------------

    best_mrr = max(
        results_table,
        key=lambda x: x["mrr"]
    )

    print("\n" + "=" * 70)
    print("BEST RESULTS")
    print("=" * 70)

    print(
        f"Best F1 threshold  : "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"F1                 : "
        f"{best_f1['f1']:.4f}"
    )

    print(
        f"Best MRR threshold : "
        f"{best_mrr['threshold']:.2f}"
    )

    print(
        f"MRR                : "
        f"{best_mrr['mrr']:.4f}"
    )

    print("\nIMPORTANT:")
    print(
        "This benchmark evaluates CrossEncoder "
        "rerank_score, not FAISS score."
    )

    print(
        "Do NOT change MIN_RETRIEVAL_SCORE yet."
    )

    print(
        "Use these results to decide whether a "
        "reranker-based evidence gate is justified."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()