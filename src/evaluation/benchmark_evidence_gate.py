import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker

DATASET_FILE = Path("data/hindi_dev.parquet")
OUTPUT_STATS_FILE = Path("data/evidence_gate_calibration.json")
CANDIDATE_K = 15
THRESHOLDS = [-1.0, -0.5, 0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def main():
    if not DATASET_FILE.exists():
        print(f"Error: Dataset not found at {DATASET_FILE}")
        return

    print("=" * 80)
    print("PHASE 4: EVIDENCE GATE THRESHOLD CALIBRATION")
    print("=" * 80)

    print("Loading dataset...")
    df = pd.read_parquet(
        DATASET_FILE,
        columns=["query_id", "query", "Answer", "passages"]
    )

    is_unanswerable_mask = df["Answer"].astype(str).str.contains(
        "No Answer Present|कोई उत्तर नहीं मिला",
        case=False,
        regex=True
    )

    df_unanswerable = df[is_unanswerable_mask]
    df_answerable_all = df[~is_unanswerable_mask]
    df_answerable = df_answerable_all[
        df_answerable_all["passages"].apply(
            lambda x: 1 in list(x["is_selected"])
        )
    ]

    total_ans = len(df_answerable)
    total_unans = len(df_unanswerable)
    total_all = total_ans + total_unans

    print(f"Total Dataset Rows       : {len(df)}")
    print(f"Answerable Queries (GT)  : {total_ans}")
    print(f"Unanswerable Queries     : {total_unans}")
    print(f"Total Evaluated Queries  : {total_all}")

    print("\nLoading Retriever and Reranker on CUDA...")
    retriever = Retriever()
    reranker = Reranker()

    # Warmup
    warmup_cands = retriever.search("परीक्षण", top_k=CANDIDATE_K)
    reranker.rerank("परीक्षण", warmup_cands, top_k=5)

    print("\n[1/2] Processing 3,037 Answerable Queries...")
    ans_records = []
    ans_max_scores = []
    ans_rel_scores = []

    for i, (_, row) in enumerate(df_answerable.iterrows(), start=1):
        query_id = int(row["query_id"])
        query = row["query"]
        passages = row["passages"]

        selected_passage_ids = {
            pid
            for pid, flag in enumerate(passages["is_selected"])
            if int(flag) == 1
        }

        candidates = retriever.search(query, top_k=CANDIDATE_K)
        results = reranker.rerank(query, candidates, top_k=5)

        max_score = float(results[0]["rerank_score"]) if results else float("-inf")
        ans_max_scores.append(max_score)

        # Check relevance of all retrieved results
        results_with_rel = []
        has_relevant = False
        max_rel_score = float("-inf")

        for res in results:
            ret_qid = int(res["query_id"])
            ret_pid = int(res["passage_id"])
            score = float(res.get("rerank_score", float("-inf")))
            is_rel = (ret_qid == query_id and ret_pid in selected_passage_ids)
            if is_rel:
                has_relevant = True
                if score > max_rel_score:
                    max_rel_score = score
            results_with_rel.append({
                "score": score,
                "is_rel": is_rel,
            })

        if max_rel_score > float("-inf"):
            ans_rel_scores.append(max_rel_score)

        ans_records.append({
            "query_id": query_id,
            "max_score": max_score,
            "results": results_with_rel,
            "has_relevant": has_relevant,
        })

        if i % 600 == 0 or i == total_ans:
            print(f"  Processed {i}/{total_ans} answerable queries ({i/total_ans*100:.1f}%)...")

    print("\n[2/2] Processing 1,959 Unanswerable Queries...")
    unans_records = []
    unans_max_scores = []

    for i, (_, row) in enumerate(df_unanswerable.iterrows(), start=1):
        query_id = int(row["query_id"])
        query = row["query"]

        candidates = retriever.search(query, top_k=CANDIDATE_K)
        results = reranker.rerank(query, candidates, top_k=5)

        max_score = float(results[0]["rerank_score"]) if results else float("-inf")
        unans_max_scores.append(max_score)

        unans_records.append({
            "query_id": query_id,
            "max_score": max_score,
        })

        if i % 500 == 0 or i == total_unans:
            print(f"  Processed {i}/{total_unans} unanswerable queries ({i/total_unans*100:.1f}%)...")

    # Score Distribution Analysis
    ans_arr = np.array(ans_max_scores)
    unans_arr = np.array(unans_max_scores)

    score_dist = {
        "answerable_max_scores": {
            "min": float(np.min(ans_arr)),
            "mean": float(np.mean(ans_arr)),
            "median": float(np.median(ans_arr)),
            "p25": float(np.percentile(ans_arr, 25)),
            "p75": float(np.percentile(ans_arr, 75)),
            "p90": float(np.percentile(ans_arr, 90)),
            "max": float(np.max(ans_arr)),
        },
        "unanswerable_max_scores": {
            "min": float(np.min(unans_arr)),
            "mean": float(np.mean(unans_arr)),
            "median": float(np.median(unans_arr)),
            "p25": float(np.percentile(unans_arr, 25)),
            "p75": float(np.percentile(unans_arr, 75)),
            "p90": float(np.percentile(unans_arr, 90)),
            "max": float(np.max(unans_arr)),
        },
    }

    # Sweep Thresholds
    threshold_results = []

    for t in THRESHOLDS:
        # Answerable analysis:
        # Grounded coverage (True Positive): Top result passing threshold is relevant
        # False Reject: No relevant result passes threshold (abstained or only irrelevant passed)
        covered_count = 0
        ans_abstain_or_miss = 0

        for rec in ans_records:
            # Filter evidence by threshold t
            passing = [r for r in rec["results"] if r["score"] >= t]
            if passing and passing[0]["is_rel"]:
                # Top evidence passing threshold is ground-truth relevant
                covered_count += 1
            else:
                ans_abstain_or_miss += 1

        ans_coverage_pct = (covered_count / total_ans) * 100
        ans_false_reject_pct = (ans_abstain_or_miss / total_ans) * 100

        # Unanswerable analysis:
        # True Abstain: All results < t (max_score < t)
        # False Accept: max_score >= t
        unans_abstain_count = sum(1 for rec in unans_records if rec["max_score"] < t)
        unans_accept_count = total_unans - unans_abstain_count

        unans_abstain_pct = (unans_abstain_count / total_unans) * 100
        unans_false_accept_pct = (unans_accept_count / total_unans) * 100

        # Overall gate accuracy
        correct_decisions = covered_count + unans_abstain_count
        gate_accuracy_pct = (correct_decisions / total_all) * 100

        threshold_results.append({
            "threshold": t,
            "answerable_coverage_pct": ans_coverage_pct,
            "answerable_false_reject_pct": ans_false_reject_pct,
            "unanswerable_abstain_pct": unans_abstain_pct,
            "unanswerable_false_accept_pct": unans_false_accept_pct,
            "gate_accuracy_pct": gate_accuracy_pct,
            "covered_count": covered_count,
            "unans_abstain_count": unans_abstain_count,
        })

    # Print Clean Table
    print("\n" + "=" * 90)
    print("EVIDENCE GATE THRESHOLD CALIBRATION RESULTS")
    print("=" * 90)
    print(f"{'Threshold':<12}{'Ans Coverage':<18}{'False Reject':<16}{'Unans Abstain':<18}{'False Accept':<16}{'Gate Accuracy':<15}")
    print("-" * 90)

    for r in threshold_results:
        print(
            f"{r['threshold']:<12.1f}"
            f"{r['answerable_coverage_pct']:<18.2f}%"
            f"{r['answerable_false_reject_pct']:<16.2f}%"
            f"{r['unanswerable_abstain_pct']:<18.2f}%"
            f"{r['unanswerable_false_accept_pct']:<16.2f}%"
            f"{r['gate_accuracy_pct']:<15.2f}%"
        )

    print("=" * 90)

    print("\nSCORE DISTRIBUTIONS (MAX RERANK SCORE PER QUERY):")
    print("-" * 80)
    print(f"{'Dataset Group':<20}{'Min':<10}{'P25':<10}{'Median':<10}{'Mean':<10}{'P75':<10}{'P90':<10}{'Max':<10}")
    print("-" * 80)
    for name, stats in [("Answerable (3037)", score_dist["answerable_max_scores"]), ("Unanswerable (1959)", score_dist["unanswerable_max_scores"])]:
        print(
            f"{name:<20}"
            f"{stats['min']:<10.2f}"
            f"{stats['p25']:<10.2f}"
            f"{stats['median']:<10.2f}"
            f"{stats['mean']:<10.2f}"
            f"{stats['p75']:<10.2f}"
            f"{stats['p90']:<10.2f}"
            f"{stats['max']:<10.2f}"
        )
    print("=" * 80)

    output_data = {
        "dataset_summary": {
            "total_rows": len(df),
            "answerable_queries": total_ans,
            "unanswerable_queries": total_unans,
            "candidate_k": CANDIDATE_K,
        },
        "score_distributions": score_dist,
        "threshold_calibration_table": threshold_results,
    }

    OUTPUT_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nCalibration data saved to: {OUTPUT_STATS_FILE}")


if __name__ == "__main__":
    main()
