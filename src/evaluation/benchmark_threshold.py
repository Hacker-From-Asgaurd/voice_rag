import sys
import statistics
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

# Experimentally validated configuration:
# E5 FAISS Top-10 -> CrossEncoder -> Top-10
INITIAL_TOP_K = 10
FINAL_TOP_K = 10

# Thresholds to evaluate.
THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.82,
    0.84,
    0.85,
    0.86,
    0.88,
    0.90,
]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("RETRIEVAL EVIDENCE THRESHOLD BENCHMARK")
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

    print("\nLoading reranker...")

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
    # Evaluate each threshold
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD RESULTS")
    print("=" * 70)

    results_table = []

    for threshold in THRESHOLDS:

        queries_with_evidence = 0
        total_evidence = 0

        relevant_retrieved = 0
        relevant_available = 0

        false_evidence = 0

        reciprocal_rank_total = 0.0

        for query_id, retrieved in all_results.items():

            relevant_passages = ground_truth[
                query_id
            ]

            # -------------------------------------------------
            # Apply FAISS evidence threshold.
            #
            # IMPORTANT:
            # result["score"] is the original FAISS score.
            # rerank_score is NOT used for this threshold.
            # -------------------------------------------------

            evidence = [
                result
                for result in retrieved
                if float(result["score"])
                >= threshold
            ]

            if evidence:

                queries_with_evidence += 1

            total_evidence += len(
                evidence
            )

            # -------------------------------------------------
            # Count available relevant passages
            # -------------------------------------------------

            relevant_available += len(
                relevant_passages
            )

            # -------------------------------------------------
            # Evaluate evidence
            # -------------------------------------------------

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

                    relevant_retrieved += 1

                else:

                    false_evidence += 1

            # -------------------------------------------------
            # Reciprocal rank of first relevant evidence
            # -------------------------------------------------

            for rank, is_relevant in enumerate(
                relevance,
                start=1
            ):

                if is_relevant:

                    reciprocal_rank_total += (
                        1.0 / rank
                    )

                    break

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

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

        if relevant_available > 0:

            evidence_recall = (
                relevant_retrieved
                /
                relevant_available
            )

        else:

            evidence_recall = 0.0

        if total_evidence > 0:

            evidence_precision = (
                relevant_retrieved
                /
                total_evidence
            )

        else:

            evidence_precision = 0.0

        mrr = (
            reciprocal_rank_total
            /
            num_queries
        )

        results_table.append({
            "Threshold": threshold,
            "Queries with evidence":
                queries_with_evidence,
            "Abstention rate":
                abstention_rate,
            "Total evidence":
                total_evidence,
            "Relevant evidence":
                relevant_retrieved,
            "False evidence":
                false_evidence,
            "Evidence precision":
                evidence_precision,
            "Evidence recall":
                evidence_recall,
            "MRR":
                mrr,
        })

        # -----------------------------------------------------
        # Print result
        # -----------------------------------------------------

        print("\n" + "-" * 70)

        print(
            f"THRESHOLD: {threshold:.2f}"
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
            f"{relevant_retrieved}"
        )

        print(
            f"False evidence         : "
            f"{false_evidence}"
        )

        print(
            f"Evidence precision     : "
            f"{evidence_precision:.4f}"
        )

        print(
            f"Evidence recall        : "
            f"{evidence_recall:.4f}"
        )

        print(
            f"MRR                    : "
            f"{mrr:.4f}"
        )

    # -----------------------------------------------------
    # Summary table
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
        f"{'MRR':<12}"
    )

    print("-" * 70)

    for row in results_table:

        print(
            f"{row['Threshold']:<12.2f}"
            f"{row['Queries with evidence']:<12}"
            f"{row['Abstention rate']:<12.3f}"
            f"{row['Evidence precision']:<12.3f}"
            f"{row['Evidence recall']:<12.3f}"
            f"{row['MRR']:<12.3f}"
        )

    # -----------------------------------------------------
    # Best threshold by F1-like balance
    # -----------------------------------------------------

    best_threshold = None
    best_score = -1.0

    for row in results_table:

        precision = row[
            "Evidence precision"
        ]

        recall = row[
            "Evidence recall"
        ]

        if precision + recall == 0:

            f1 = 0.0

        else:

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

        if f1 > best_score:

            best_score = f1
            best_threshold = row[
                "Threshold"
            ]

    print("\n" + "=" * 70)
    print("RECOMMENDED THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"Evidence F1: {best_score:.4f}"
    )

    print("=" * 70)

    print("\nIMPORTANT:")
    print(
        "This benchmark evaluates the FAISS similarity "
        "threshold, not the CrossEncoder rerank score."
    )

    print(
        "The selected threshold should be validated "
        "again with end-to-end generation before "
        "changing production configuration."
    )


if __name__ == "__main__":
    main()