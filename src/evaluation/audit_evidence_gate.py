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
OUTPUT_STATS_FILE = Path("data/evidence_gate_audit.json")
CANDIDATE_K = 15
THRESHOLDS = [-1.0, -0.5, 0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def calc_stats(arr):
    if len(arr) == 0:
        return {"min": 0, "p25": 0, "median": 0, "mean": 0, "p75": 0, "p90": 0, "max": 0}
    a = np.array(arr)
    return {
        "min": float(np.min(a)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": float(np.max(a)),
    }


def main():
    if not DATASET_FILE.exists():
        print(f"Error: Dataset not found at {DATASET_FILE}")
        return

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

    retriever = Retriever()
    reranker = Reranker()

    # Warmup
    warmup_cands = retriever.search("परीक्षण", top_k=CANDIDATE_K)
    reranker.rerank("परीक्षण", warmup_cands, top_k=5)

    # 1. Process Answerable queries
    group1_success = []      # queries with at least one relevant passage in top 5
    group2_failure = []      # queries with NO relevant passage in top 5
    true_relevant_scores = []
    ans_distractor_scores = []

    for i, (_, row) in enumerate(df_answerable.iterrows(), start=1):
        query_id = int(row["query_id"])
        query = row["query"]
        passages = row["passages"]

        selected_passages = {
            pid
            for pid, flag in enumerate(passages["is_selected"])
            if int(flag) == 1
        }

        candidates = retriever.search(query, top_k=CANDIDATE_K)
        results = reranker.rerank(query, candidates, top_k=5)

        rel_scores = []
        all_scores = []

        for res in results:
            ret_qid = int(res["query_id"])
            ret_pid = int(res["passage_id"])
            score = float(res.get("rerank_score", float("-inf")))
            all_scores.append(score)

            is_rel = (ret_qid == query_id and ret_pid in selected_passages)
            if is_rel:
                rel_scores.append(score)
                true_relevant_scores.append(score)

        if rel_scores:
            # Group 1: Retrieval Success @5
            group1_success.append({
                "query_id": query_id,
                "max_rel_score": max(rel_scores),
                "all_scores": all_scores,
            })
        else:
            # Group 2: Retrieval Failure @5
            max_distractor = max(all_scores) if all_scores else float("-inf")
            ans_distractor_scores.append(max_distractor)
            group2_failure.append({
                "query_id": query_id,
                "max_distractor_score": max_distractor,
                "all_scores": all_scores,
            })

    # 2. Process Unanswerable queries (Group 3)
    unans_distractor_scores = []
    group3_unans = []

    for i, (_, row) in enumerate(df_unanswerable.iterrows(), start=1):
        query_id = int(row["query_id"])
        query = row["query"]

        candidates = retriever.search(query, top_k=CANDIDATE_K)
        results = reranker.rerank(query, candidates, top_k=5)

        all_scores = [float(res.get("rerank_score", float("-inf"))) for res in results]
        max_score = max(all_scores) if all_scores else float("-inf")
        unans_distractor_scores.append(max_score)

        group3_unans.append({
            "query_id": query_id,
            "max_score": max_score,
            "all_scores": all_scores,
        })

    num_success = len(group1_success)
    num_failure = len(group2_failure)

    # 3. Calculate Threshold Metrics
    threshold_table = []

    for t in THRESHOLDS:
        # Retention: % of Group 1 where max_rel_score >= t
        retained = sum(1 for q in group1_success if q["max_rel_score"] >= t)
        retention_pct = (retained / num_success * 100) if num_success > 0 else 0.0

        # Distractor Reject: % of Group 2 where all scores < t
        distractor_rejected = sum(1 for q in group2_failure if q["max_distractor_score"] < t)
        distractor_reject_pct = (distractor_rejected / num_failure * 100) if num_failure > 0 else 0.0

        # Unans Abstain: % of Group 3 where all scores < t
        unans_abstained = sum(1 for q in group3_unans if q["max_score"] < t)
        unans_abstain_pct = (unans_abstained / total_unans * 100) if total_unans > 0 else 0.0

        # Supported Coverage: % of all 3037 answerable queries where GT in top 5 AND rel_score >= t
        supported_cov_pct = (retained / total_ans * 100) if total_ans > 0 else 0.0

        threshold_table.append({
            "threshold": t,
            "retention_pct": retention_pct,
            "distractor_reject_pct": distractor_reject_pct,
            "unans_abstain_pct": unans_abstain_pct,
            "supported_cov_pct": supported_cov_pct,
            "retained_count": retained,
            "distractor_rejected_count": distractor_rejected,
            "unans_abstained_count": unans_abstained,
        })

    stats_rel = calc_stats(true_relevant_scores)
    stats_ans_dist = calc_stats(ans_distractor_scores)
    stats_unans_dist = calc_stats(unans_distractor_scores)

    # 4. Print Compact Summary
    print("=" * 60)
    print("PHASE 4.1: EVIDENCE GATE CALIBRATION AUDIT")
    print("=" * 60)
    print("\nDataset")
    print(f"  Answerable                  : {total_ans}")
    print(f"  Retrieval success @5        : {num_success} ({num_success/total_ans*100:.2f}%)")
    print(f"  Retrieval failure @5        : {num_failure} ({num_failure/total_ans*100:.2f}%)")
    print(f"  Unanswerable                : {total_unans}")
    print()
    print(f"{'Threshold':<16}{'Retention':<17}{'Distractor Reject':<21}{'Unans Abstain':<17}{'Supported Coverage':<18}")
    print("-" * 87)

    for r in threshold_table:
        print(
            f"{r['threshold']:<16.1f}"
            f"{r['retention_pct']:<17.2f}%"
            f"{r['distractor_reject_pct']:<21.2f}%"
            f"{r['unans_abstain_pct']:<17.2f}%"
            f"{r['supported_cov_pct']:<18.2f}%"
        )
    print("-" * 87)

    print("\nSCORE DISTRIBUTIONS")
    print("-" * 80)
    print(f"{'Distribution':<28}{'Min':<9}{'P25':<9}{'Median':<9}{'Mean':<9}{'P75':<9}{'P90':<9}{'Max':<9}")
    print("-" * 80)
    for name, st in [
        ("True relevant passages", stats_rel),
        ("Answerable distractors", stats_ans_dist),
        ("Unanswerable distractors", stats_unans_dist),
    ]:
        print(
            f"{name:<28}"
            f"{st['min']:<9.2f}"
            f"{st['p25']:<9.2f}"
            f"{st['median']:<9.2f}"
            f"{st['mean']:<9.2f}"
            f"{st['p75']:<9.2f}"
            f"{st['p90']:<9.2f}"
            f"{st['max']:<9.2f}"
        )
    print("=" * 80)

    # 5. Save JSON
    output_data = {
        "dataset": {
            "answerable_total": total_ans,
            "retrieval_success_at_5": num_success,
            "retrieval_success_pct": round(num_success / total_ans * 100, 2),
            "retrieval_failure_at_5": num_failure,
            "retrieval_failure_pct": round(num_failure / total_ans * 100, 2),
            "unanswerable_total": total_unans,
            "candidate_k": CANDIDATE_K,
        },
        "threshold_calibration": threshold_table,
        "score_distributions": {
            "true_relevant_passages": stats_rel,
            "answerable_distractors": stats_ans_dist,
            "unanswerable_distractors": stats_unans_dist,
        },
    }

    OUTPUT_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nAudit data saved to: {OUTPUT_STATS_FILE}")


if __name__ == "__main__":
    main()
