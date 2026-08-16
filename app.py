import os
import sys
import gradio as gr

# Ensure root and src are in Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from speech.voice_pipeline import VoiceRAGPipeline
from harness.guardrails import UNSAFE_RESPONSE
from harness.pipeline import ABSTENTION_TEXT

# Initialize Voice RAG Pipeline
print("Initializing Voice RAG Pipeline for Hugging Face Spaces...")
voice_pipeline = VoiceRAGPipeline()
print("Voice RAG Pipeline ready.")

def process_query(audio_file, text_input):
    if audio_file is not None:
        result = voice_pipeline.run(audio_source=audio_file, language_code="unknown")
        query_text = result.get("transcript", "")
    elif text_input and text_input.strip():
        result = voice_pipeline.query_text(text_input.strip())
        query_text = text_input.strip()
    else:
        return "", "Please speak into the microphone or type a question.", "Awaiting input", "No sources", ""

    answer = result.get("answer", "")
    grounded = result.get("grounded", False)
    sources = result.get("sources", [])
    lat = result.get("latency", {})

    if answer == UNSAFE_RESPONSE:
        status = "🔴 BLOCKED (Safety Guardrail)"
    elif grounded:
        status = "🟢 GROUNDED (Evidence Backed)"
    else:
        status = "🟡 SAFE ABSTENTION"

    # Format sources
    sources_text = ""
    for idx, s in enumerate(sources, 1):
        sources_text += f"### Source {idx:02d} (Score: {s.get('rerank_score', s.get('score', 0)):.2f})\n{s.get('chunk', '')}\n\n---\n"
    if not sources_text:
        sources_text = "No supporting passages retained by evidence gate."

    # Format latency telemetry
    lat_text = (
        f"**Live Single-Request Latency:**\n"
        f"- Speech-to-Text (Sarvam Saaras v3): {lat.get('stt_ms', 0):.1f} ms\n"
        f"- E5 Dense Search (k=15): {lat.get('retrieval_ms', 0):.1f} ms\n"
        f"- CrossEncoder Reranking (Top-5): {lat.get('reranker_ms', 0):.1f} ms\n"
        f"- Evidence Gate (T=0.80): {lat.get('evidence_gate_ms', 0):.1f} ms\n"
        f"- Gemini Generation: {lat.get('generation_ms', 0):.1f} ms\n"
        f"- **Total End-to-End**: {lat.get('total_ms', 0):.1f} ms\n\n"
        f"**Offline 3,037-Query Benchmark (k=15):**\n"
        f"- Recall@1: 34.84% | Recall@5: 71.78% | MRR: 0.4902\n"
        f"- P50: 87.6 ms | P95: 142.9 ms | P100: 369.3 ms"
    )

    return query_text, answer, status, sources_text, lat_text

# Custom CSS for dark technical styling
custom_css = """
body { background-color: #080c14; color: #f3f4f6; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css, title="VOICE RAG — HH Goa 2026") as demo:
    gr.Markdown("# 🎙️ VOICE RAG — HH GOA 2026\n### Multilingual Voice-Enabled Grounded Retrieval Dashboard (MSMARCO-XI Hindi)")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voice Input (Hindi / English / Marathi / Hinglish)")
            text_input = gr.Textbox(placeholder="Or type a question (e.g. मैनहट्टन परियोजना क्या थी?)", label="Text Query Fallback")
            submit_btn = gr.Button("Search / Transcribe", variant="primary")
            status_box = gr.Textbox(label="Grounding Status", interactive=False)
            transcript_box = gr.Textbox(label="Detected Transcription / Query", interactive=False)

        with gr.Column(scale=1):
            answer_box = gr.Textbox(label="Generated Grounded Answer", lines=4, interactive=False)
            latency_box = gr.Markdown(label="Latency Telemetry & Benchmark")

    with gr.Row():
        sources_box = gr.Markdown(label="Retrieved Evidence Sources")

    submit_btn.click(
        fn=process_query,
        inputs=[audio_input, text_input],
        outputs=[transcript_box, answer_box, status_box, sources_box, latency_box],
    )

if __name__ == "__main__":
    demo.launch()
