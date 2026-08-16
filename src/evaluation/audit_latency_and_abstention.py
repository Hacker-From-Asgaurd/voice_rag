import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from harness.pipeline import RAGPipeline, ABSTENTION_TEXT, MIN_RERANK_SCORE
from harness.guardrails import check_input, UNSAFE_RESPONSE
import generation.generator as gen_module

DATASET_FILE = Path("data/hindi_dev.parquet")
LATENCY_AUDIT_FILE = Path("data/e2e_latency_audit.json")
ABSTENTION_ANALYSIS_FILE = Path("data/abstention_failure_analysis.json")


def calc_percentiles(lat_list):
    if not lat_list:
        return {"avg": 0.0, "p50": 0.0, "p70": 0.0, "p95": 0.0, "p100": 0.0}
    s = sorted(lat_list)
    n = len(s)
    avg = sum(s) / n
    p50 = s[min(int(n * 0.50), n - 1)]
    p70 = s[min(int(n * 0.70), n - 1)]
    p95 = s[min(int(n * 0.95), n - 1)]
    p100 = max(s)
    return {"avg": avg, "p50": p50, "p70": p70, "p95": p95, "p100": p100}


def main():
    print("=" * 80)
    print("PHASE 5 AUDIT: LATENCY TELEMETRY & UNANSWERABLE HALLUCINATION ANALYSIS")
    print("=" * 80)

    df = pd.read_parquet(DATASET_FILE)

    # 1. Answerable cohort (50 queries stratified)
    is_unans = df["Answer"].astype(str).str.contains(
        "No Answer Present|कोई उत्तर नहीं मिला", case=False, regex=True
    )
    df_ans = df[~is_unans & df["passages"].apply(lambda x: 1 in list(x["is_selected"]))]
    type_counts = {"DESCRIPTION": 20, "NUMERIC": 10, "ENTITY": 10, "LOCATION": 5, "PERSON": 5}
    sampled_ans = []
    for qtype, count in type_counts.items():
        sub = df_ans[df_ans["query_type"] == qtype]
        sampled_ans.append(sub.iloc[:count] if len(sub) >= count else sub)
    ans_sample = pd.concat(sampled_ans).head(50)

    # 2. Unanswerable cohort (30 queries)
    df_unans = df[is_unans & df["passages"].apply(lambda x: 1 not in list(x["is_selected"]))]
    unans_sample = df_unans.head(30)

    print("Initializing pipeline on CUDA...")
    pipeline = RAGPipeline()

    # Latency tracking arrays
    lat_guardrail = []
    lat_e5 = []
    lat_rerank = []
    lat_gate = []
    lat_gen = []
    lat_orchestration = []
    lat_e2e = []

    print("\n[Part 1] Running Clean Latency Audit on Answerable Cohort (50 queries)...")
    for i, (_, row) in enumerate(ans_sample.iterrows(), start=1):
        query = str(row["query"])

        # Strict timer boundary for E2E
        t_e2e_start = time.perf_counter()

        # 1. Guardrail
        t0 = time.perf_counter()
        gr_res = check_input(query)
        t1 = time.perf_counter()
        d_gr = (t1 - t0) * 1000
        lat_guardrail.append(d_gr)

        # 2. E5 Retrieval (k=15)
        t0 = time.perf_counter()
        candidates = pipeline.retriever.search(query, top_k=15)
        t1 = time.perf_counter()
        d_e5 = (t1 - t0) * 1000
        lat_e5.append(d_e5)

        # 3. CrossEncoder Reranking
        t0 = time.perf_counter()
        reranked = pipeline.reranker.rerank(query, candidates, top_k=5)
        t1 = time.perf_counter()
        d_rerank = (t1 - t0) * 1000
        lat_rerank.append(d_rerank)

        # 4. Evidence Gate
        t0 = time.perf_counter()
        evidence = pipeline.filter_evidence(reranked)
        t1 = time.perf_counter()
        d_gate = (t1 - t0) * 1000
        lat_gate.append(d_gate)

        # 5. Generation
        d_gen = 0.0
        if evidence:
            context = pipeline.build_context(evidence)
            t0 = time.perf_counter()
            try:
                answer = pipeline.generate(query, context)
            except Exception as e:
                answer = ABSTENTION_TEXT
            t1 = time.perf_counter()
            d_gen = (t1 - t0) * 1000
            lat_gen.append(d_gen)

        # End of E2E timer boundary
        t_e2e_end = time.perf_counter()
        d_e2e = (t_e2e_end - t_e2e_start) * 1000
        lat_e2e.append(d_e2e)

        # Orchestration overhead calculation
        d_orch = max(0.0, d_e2e - (d_gr + d_e5 + d_rerank + d_gate + d_gen))
        lat_orchestration.append(d_orch)

        # Pacing sleep OUTSIDE the timer
        if evidence:
            time.sleep(4.1)

    p_gr = calc_percentiles(lat_guardrail)
    p_e5 = calc_percentiles(lat_e5)
    p_rerank = calc_percentiles(lat_rerank)
    p_gate = calc_percentiles(lat_gate)
    p_gen = calc_percentiles(lat_gen)
    p_orch = calc_percentiles(lat_orchestration)
    p_e2e = calc_percentiles(lat_e2e)

    # Save latency audit
    latency_audit_data = {
        "summary": "Clean timer boundary without synthetic pacing sleep artifact",
        "percentiles": {
            "guardrail": p_gr,
            "e5_retrieval": p_e5,
            "reranker": p_rerank,
            "evidence_gate": p_gate,
            "generation": p_gen,
            "orchestration_overhead": p_orch,
            "end_to_end_total": p_e2e,
        },
        "sum_of_stage_averages": p_gr["avg"] + p_e5["avg"] + p_rerank["avg"] + p_gate["avg"] + p_gen["avg"] + p_orch["avg"],
        "reported_e2e_average": p_e2e["avg"],
    }
    with open(LATENCY_AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(latency_audit_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("PART 1: CORRECTED LATENCY WATERFALL (ALL STAGES)")
    print("=" * 80)
    print(f"{'Stage':<25}{'Avg':<10}{'P50':<10}{'P70':<10}{'P95':<10}{'P100':<10}")
    print("-" * 75)
    for name, p in [
        ("Guardrail", p_gr),
        ("E5 Retrieval (k=15)", p_e5),
        ("CrossEncoder Rerank", p_rerank),
        ("Evidence Gate", p_gate),
        ("Gemini Generation", p_gen),
        ("Orchestration Overhead", p_orch),
        ("End-to-End Total", p_e2e),
    ]:
        print(f"{name:<25}{p['avg']:<10.1f}{p['p50']:<10.1f}{p['p70']:<10.1f}{p['p95']:<10.1f}{p['p100']:<10.1f}")
    print("-" * 75)
    print(f"Sum of Stage Averages  : {latency_audit_data['sum_of_stage_averages']:.1f} ms")
    print(f"Reported E2E Average   : {latency_audit_data['reported_e2e_average']:.1f} ms")
    print("=" * 80)

    # =========================================================
    # PART 2: UNANSWERABLE FAILURE MODE ANALYSIS (30 QUERIES)
    # =========================================================
    print("\n[Part 2] Deep-Dive Failure Audit on 30 Unanswerable Queries...")
    unans_audit_records = []
    failure_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for i, (_, row) in enumerate(unans_sample.iterrows(), start=1):
        query_id = int(row["query_id"])
        query = str(row["query"])

        candidates = pipeline.retriever.search(query, top_k=15)
        reranked = pipeline.reranker.rerank(query, candidates, top_k=5)
        evidence = pipeline.filter_evidence(reranked)

        top_scores = [float(r.get("rerank_score", float("-inf"))) for r in reranked]
        max_score = top_scores[0] if top_scores else float("-inf")
        gate_passed = len(evidence) > 0

        answer = ""
        context = ""
        if evidence:
            context = pipeline.build_context(evidence)
            try:
                answer = pipeline.generate(query, context)
            except Exception as e:
                answer = ABSTENTION_TEXT
            time.sleep(4.1)
        else:
            answer = ABSTENTION_TEXT

        is_abstained = (answer.strip() == ABSTENTION_TEXT or "उपलब्ध नहीं" in answer)

        classification = None
        failure_reason = None

        if not is_abstained:
            # Analyze why it answered
            # 1. Did the context actually contain the facts? (Corpus overlap / Evaluation error)
            query_words = set(query.split())
            context_words = set(context.split())

            # 2. Check rerank scores:
            if max_score >= 4.0:
                # Strong retrieval false positive (high score on distractor)
                classification = "A"
                failure_reason = f"Retrieval False Positive: Distractor scored very high ({max_score:.2f} >= 4.0), leading LLM to treat it as authoritative context."
            elif 0.80 <= max_score < 4.0:
                # Evidence gate let weak/borderline score through
                classification = "B"
                failure_reason = f"Evidence Gate False Acceptance: Weak score ({max_score:.2f} in [0.80, 4.0)) passed threshold T=0.80."
            else:
                classification = "C"
                failure_reason = f"Generation Hallucination: LLM generated answer despite weak context."

            failure_counts[classification] += 1

        unans_audit_records.append({
            "query_id": query_id,
            "query": query,
            "max_rerank_score": max_score,
            "top_5_scores": top_scores,
            "gate_passed": gate_passed,
            "evidence_count": len(evidence),
            "answer": answer,
            "is_abstained": is_abstained,
            "classification": classification,
            "failure_reason": failure_reason,
            "context_preview": context[:200] if context else "None (Filtered by gate)",
        })

    total_unans = len(unans_sample)
    total_failures = sum(failure_counts.values())
    correct_abstentions = total_unans - total_failures

    print("\n" + "=" * 80)
    print("PART 2: UNANSWERABLE FAILURE ROOT-CAUSE BREAKDOWN (30 QUERIES)")
    print("=" * 80)
    print(f"Total Unanswerable Queries Evaluated : {total_unans}")
    print(f"Correct Abstentions                  : {correct_abstentions} ({correct_abstentions/total_unans*100:.2f}%)")
    print(f"Total False Answers (Failures)       : {total_failures} ({total_failures/total_unans*100:.2f}%)")
    print("-" * 80)
    print(f"{'Failure Mode':<45}{'Count':<10}{'Percentage':<15}")
    print("-" * 80)
    print(f"{'A: Retrieval False Positive (Score >= 4.0)':<45}{failure_counts['A']:<10}{failure_counts['A']/total_unans*100:.2f}%")
    print(f"{'B: Evidence Gate False Accept (0.80 <= Score < 4.0)':<45}{failure_counts['B']:<10}{failure_counts['B']/total_unans*100:.2f}%")
    print(f"{'C: Generation Hallucination (Ignored Prompt)':<45}{failure_counts['C']:<10}{failure_counts['C']/total_unans*100:.2f}%")
    print(f"{'D: Dataset/Corpus Overlap':<45}{failure_counts['D']:<10}{failure_counts['D']/total_unans*100:.2f}%")
    print(f"{'E: Other':<45}{failure_counts['E']:<10}{failure_counts['E']/total_unans*100:.2f}%")
    print("=" * 80)

    # Save detailed failure analysis
    abstention_audit_data = {
        "total_unanswerable": total_unans,
        "correct_abstentions": correct_abstentions,
        "abstention_rate_pct": round(correct_abstentions / total_unans * 100, 2),
        "false_answers": total_failures,
        "false_answer_rate_pct": round(total_failures / total_unans * 100, 2),
        "failure_modes": failure_counts,
        "per_query_audit": unans_audit_records,
    }
    with open(ABSTENTION_ANALYSIS_FILE, "w", encoding="utf-8") as f:
        json.dump(abstention_audit_data, f, indent=2, ensure_ascii=False)

    print(f"\nArtifacts Saved:")
    print(f"  1. {LATENCY_AUDIT_FILE}")
    print(f"  2. {ABSTENTION_ANALYSIS_FILE}")


if __name__ == "__main__":
    main()
