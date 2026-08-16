import os
import sys
import importlib.util
import uvicorn
import gradio as gr

# Ensure root and src are in Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load FastAPI app directly from app/main.py
main_py_path = os.path.join(ROOT_DIR, "app", "main.py")
spec = importlib.util.spec_from_file_location("fastapi_main", main_py_path)
fastapi_main = importlib.util.module_from_spec(spec)
sys.modules["fastapi_main"] = fastapi_main
spec.loader.exec_module(fastapi_main)
fastapi_app = fastapi_main.app

# Mount Gradio Blocks wrapper for Hugging Face Spaces
with gr.Blocks(title="VOICE RAG — HH Goa 2026") as demo:
    pass

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
