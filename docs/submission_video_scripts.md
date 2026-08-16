# HH Goa 2026 Task 2: Submission Video Scripts & Social Media Checklist

---

## 🎬 Video 1 — Team / Process Video (Strict 90-Second Limit)

**Theme**: "How we engineered a sub-200ms multilingual retrieval core and grounded voice RAG system."  
**Focus**: Engineering process, technical tradeoffs, benchmarks, and team workflow (NOT just product demo).

| Timestamp | Visual / Screen On Camera | Spoken Script (Word-for-Word Guide) |
| :--- | :--- | :--- |
| **0:00 – 0:15** | Team introducing themselves + problem statement slide on screen. | *"Hey everyone! We are Team [Team Name] for HH Goa 2026 Task 2. Our challenge: build an end-to-end voice-enabled RAG system on the MSMARCO-XI Hindi dataset with a strict sub-200ms retrieval budget and zero tolerance for hallucinations."* |
| **0:15 – 0:35** | Screen recording showing `src/chunking.py` and the chunking ablation results (`data/duplicate_investigation.json`). | *"We started by engineering our chunking strategy. Rather than naive fixed splitting, we compared fixed, sentence-aware, and adaptive parent-child chunking across 50,000+ passages. Our duplicate investigation proved that 100% of duplicate IDs were legitimate long-document child splits, keeping 0 slot waste in the candidate pool."* |
| **0:35 – 0:55** | Screen recording showing GPU benchmark runs (`src/evaluation/full_retrieval_benchmark.py` and `benchmark_candidate_pool.py`). | *"For retrieval, we paired multilingual E5-Base with a CrossEncoder reranker. We scaled candidate pool depth from k=10 to k=30 across all 3,037 benchmark queries. k=15 proved to be our optimal Pareto knee—achieving 71.8% Recall@5 and 0.490 MRR with an average retrieval latency of just 93 ms and P95 under 143 ms, safely beating the sub-200ms target."* |
| **0:55 – 1:15** | Screen recording showing the Evidence Gate calibration audit table (`data/evidence_gate_audit.json`). | *"Next was hallucination defense. We audited evidence gating across 3,037 answerable and 1,959 unanswerable queries. By isolating true evidence from distractors, our gate retains 84.3% of retrieved ground-truth evidence while rejecting over 44% of unanswerable noise before generation."* |
| **1:15 – 1:30** | Quick shot of team looking at the UI running live + wrap-up. | *"We integrated Sarvam Saaras v3 for multilingual voice STT and Gemini for grounded answers inside a FastAPI harness with real-time latency waterfall telemetry. That was our engineering journey—see you in Goa!"* |

---

## 🎥 Video 2 — End-to-End Product Demo Video

**Theme**: "Live end-to-end walkthrough of Voice RAG: Voice input, live transcription, grounded answer, source attribution, and real-time latency waterfall."

### Demo Walkthrough Steps:

1. **Scene 1: System Overview (0:00 – 0:20)**
   - Show the browser interface at `http://127.0.0.1:8000`.
   - Point out the active status pills: `SARVAM SAARAS v3`, `E5-BASE (k=15)`, `GATE T=0.80`.
   - Highlight the **System Benchmark** panel showing the offline 3,037-query numbers (P95: 142.9 ms).

2. **Scene 2: Supported Hindi Voice Query (0:20 – 0:50)**
   - Click the central microphone button (pulse animation turns rose, waveform moves).
   - Speak in Hindi: *"मैनहट्टन परियोजना क्या थी?"* (What was the Manhattan Project?).
   - Watch the UI transition to processing $\to$ live transcript appears: `"Manhattan परियोजना क्या थी?"` with `hi-IN` badge.
   - Grounded Answer card pops up with emerald **`GROUNDED`** badge.
   - Click **`View full evidence ▾`** on Source 01 to show exact retrieved passage with score `10.85`.
   - Point to the **Live Request Latency Waterfall**: show STT (~700ms), Retrieval Core (~275ms), Generation (~1.1s).

3. **Scene 3: Cross-Lingual English Voice Query (0:50 – 1:10)**
   - Click the microphone button and ask in English: *"What was the purpose of the Manhattan Project?"*
   - Show Sarvam transcription in `en-IN`, cross-lingual E5 retrieval from Hindi corpus, and English grounded answer.

4. **Scene 4: Unanswerable Safe Abstention Query (1:10 – 1:30)**
   - Ask: *"भारत में 1950 में कितने हवाई अड्डे थे?"*
   - Show amber **`SAFE ABSTENTION`** badge appearing with response: `"जानकारी दिए गए संदर्भ में उपलब्ध नहीं है।"`.
   - Point out that 0 ungrounded passages passed the evidence gate.

5. **Scene 5: Adversarial Safety Guardrail (1:30 – 1:45)**
   - Ask or type: *"How to hack a bank account?"*.
   - Show instant sub-millisecond block (0.02 ms) with crimson **`BLOCKED`** badge and safe refusal message.
   - Conclude demo.

---

## 📢 Social Media Promotion Checklist (Mandatory)

According to competition rules, **both videos must be uploaded by EVERY team member individually** to:
- [ ] **Instagram** (at least 1 team member account must be public)
- [ ] **X (Twitter)**
- [ ] **LinkedIn**

### Mandatory Hashtag & Post Copy Template:
```text
🚀 Excited to showcase our project for HH Goa 2026 Shortlisting Task 2: Voice-Enabled Multilingual RAG Model! 

We engineered an end-to-end voice-to-grounded-answer RAG system on MSMARCO-XI:
🎙️ Sarvam AI Saaras v3 Multilingual STT (Hindi, English, Marathi, Code-Mixed)
⚡ Sub-200ms Retrieval Core (E5-Base + CrossEncoder k=15 with 71.8% Recall@5)
🛡️ Calibrated Evidence Gate & Hallucination Guardrails
📊 Real-time Latency Waterfall Telemetry & Source Attribution

Check out our full architecture and demo!

#RAGInGoa #VoiceRAG #AI4Bharat #HHGoa2026 #GenerativeAI #RAG
```
