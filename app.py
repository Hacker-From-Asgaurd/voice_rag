import os
import sys
import gradio as gr

# Ensure root and src are in Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

from speech.voice_pipeline import VoiceRAGPipeline
from harness.guardrails import UNSAFE_RESPONSE

print("Initializing Voice RAG Pipeline...")
voice_pipeline = VoiceRAGPipeline()
print("Voice RAG Pipeline ready.")

def get_field(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def core_process(audio_file, text_input):
    # Extract file path safely from Gradio audio object
    audio_path = None
    if isinstance(audio_file, dict):
        audio_path = audio_file.get("path") or audio_file.get("name")
    elif isinstance(audio_file, str) and audio_file.strip():
        audio_path = audio_file.strip()
    elif isinstance(audio_file, tuple):
        import tempfile, soundfile as sf
        sr, y = audio_file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, sr)
            audio_path = f.name

    if audio_path and os.path.exists(audio_path):
        result = voice_pipeline.run(audio_source=audio_path, language_code="unknown")
        query_text = get_field(result, "transcript", "")
    elif text_input and str(text_input).strip():
        result = voice_pipeline.query_text(str(text_input).strip())
        query_text = str(text_input).strip()
    else:
        return (
            "", 
            "Please record audio or type a question to run Voice RAG.", 
            "⚪ STANDBY", 
            "No evidence sources loaded.",
            "Awaiting query execution..."
        )

    answer = get_field(result, "answer", "")
    grounded = get_field(result, "grounded", False)
    sources = get_field(result, "sources", [])
    lat = get_field(result, "latency", None)

    stt_ms = get_field(lat, "stt_ms", 0.0) or 0.0
    guardrail_ms = get_field(lat, "guardrail_ms", 0.1) or 0.1
    retrieval_ms = get_field(lat, "retrieval_ms", 0.0) or 0.0
    reranker_ms = get_field(lat, "reranker_ms", 0.0) or 0.0
    evidence_gate_ms = get_field(lat, "evidence_gate_ms", 0.1) or 0.1
    generation_ms = get_field(lat, "generation_ms", 0.0) or 0.0
    total_ms = get_field(lat, "total_ms", 0.0) or 0.0
    core_ms = retrieval_ms + reranker_ms

    if answer == UNSAFE_RESPONSE:
        status = "🔴 BLOCKED (Safety Guardrail Refusal)"
    elif grounded:
        status = "🟢 GROUNDED (Evidence-Backed Answer)"
    else:
        status = "🟡 SAFE ABSTENTION (Unanswerable / Low Confidence)"

    # Format Source Attribution Cards
    if sources:
        sources_md = "### 📑 Retrieved Evidence Sources (Top Passages)\n\n"
        for idx, s in enumerate(sources, 1):
            score_val = get_field(s, "rerank_score") or get_field(s, "score") or 0.0
            score_badge = f"`Score: {score_val:.2f}`" if score_val else ""
            q_id = get_field(s, "query_id", "N/A")
            p_id = get_field(s, "passage_id", "N/A")
            raw_chunk = get_field(s, "chunk", "")
            chunk_text = str(raw_chunk).replace("\n", " ")
            sources_md += f"**Source {idx:02d}** {score_badge} `Query #{q_id} Passage #{p_id}`\n\n> {chunk_text}\n\n---\n"
    else:
        sources_md = "*(No supporting evidence passages met the T=0.80 calibrated evidence gate threshold)*"

    # Format Latency Breakdown & Official Benchmark
    lat_md = f"""
### ⚡ Latency Telemetry Breakdown

| Stage | Measured Latency | Budget / Target |
| :--- | :---: | :---: |
| **Speech-to-Text (Sarvam Saaras v3)** | `{stt_ms:.1f} ms` | Cloud STT API |
| **Input Safety Guardrail** | `{guardrail_ms:.1f} ms` | < 1 ms |
| **E5 Dense Search (k=15)** | `{retrieval_ms:.1f} ms` | < 120 ms |
| **CrossEncoder Reranker (Top-5)** | `{reranker_ms:.1f} ms` | < 60 ms |
| **Evidence Gate Filter (T=0.80)** | `{evidence_gate_ms:.1f} ms` | < 1 ms |
| **👉 LIVE RETRIEVAL CORE TOTAL** | **`{core_ms:.1f} ms`** | **< 200 ms** ⚡ |
| **Gemini LLM Generation** | `{generation_ms:.1f} ms` | Cloud LLM |
| **⏱️ TOTAL END-TO-END TURNAROUND** | **`{total_ms:.1f} ms`** | Full Voice Loop |

---

### 🏆 Standardized 3,037-Query Benchmark (MSMARCO-XI Hindi)
- **Recall@1**: `34.84%` | **Recall@5**: `71.78%` | **MRR**: `0.4902`
- **P50 Latency**: `87.61 ms` | **P95 Latency**: `142.92 ms` | **P100 (Max)**: `369.32 ms`
"""

    return query_text, answer, status, sources_md, lat_md

# Top-level ZeroGPU decorator (duration=5s uses minimal quota per query)
if has_spaces:
    @spaces.GPU(duration=5)
    def process_query(audio_file, text_input):
        return core_process(audio_file, text_input)
else:
    def process_query(audio_file, text_input):
        return core_process(audio_file, text_input)

custom_css = """
body { background-color: #080c14 !important; color: #f3f4f6 !important; font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 24px !important; background-color: #080c14 !important; }
.dark, .gradio-container { background-color: #080c14 !important; }
h1, h2, h3 { color: #38bdf8 !important; font-weight: 700 !important; }
.primary-btn { background: linear-gradient(135deg, #0284c7, #6366f1) !important; color: white !important; font-weight: 600 !important; border: none !important; border-radius: 8px !important; height: 48px !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css, title="VOICE RAG — HH Goa 2026") as demo:
    gr.Markdown(
        "# 🎙️ VOICE RAG — HH GOA 2026\n"
        "**Multilingual Voice-Enabled Grounded RAG Pipeline** · `Sarvam Saaras v3` · `Multilingual E5-Base (k=15)` · `mMARCO CrossEncoder` · `Gate T=0.80`\n"
        "Supports **Hindi**, **English**, **Marathi**, and **Code-Mixed (Hinglish)** spoken queries across 50,311 MSMARCO-XI Hindi passages."
    )
    
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 🎤 Voice & Text Input")
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Voice Input (Speak into microphone)",
            )
            text_input = gr.Textbox(
                placeholder="Or type your question here (e.g. मैनहट्टन परियोजना क्या थी? / What is Manhattan Project?)", 
                label="Text Query Fallback",
                lines=1,
            )
            submit_btn = gr.Button("⚡ Run Voice RAG Query", variant="primary", elem_classes=["primary-btn"])
            
            status_box = gr.Textbox(label="🛡️ Grounding & Guardrail Status", interactive=False)
            transcript_box = gr.Textbox(label="📝 Speech Transcription / Query", interactive=False)

        with gr.Column(scale=6):
            gr.Markdown("### 🤖 Grounded Generation & Latency Waterfall")
            answer_box = gr.Textbox(label="Grounded Output Answer", lines=5, interactive=False)
            latency_box = gr.Markdown()

    with gr.Row():
        sources_box = gr.Markdown()

    # Event handlers
    submit_btn.click(
        fn=process_query,
        inputs=[audio_input, text_input],
        outputs=[transcript_box, answer_box, status_box, sources_box, latency_box],
        api_name=False
    )
    text_input.submit(
        fn=process_query,
        inputs=[audio_input, text_input],
        outputs=[transcript_box, answer_box, status_box, sources_box, latency_box],
        api_name=False
    )
    audio_input.change(
        fn=process_query,
        inputs=[audio_input, text_input],
        outputs=[transcript_box, answer_box, status_box, sources_box, latency_box],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(show_api=False)
