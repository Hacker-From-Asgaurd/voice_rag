import os
import sys
import importlib.util
import gradio as gr

# Ensure root and src are in Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load the exact FastAPI app from app/main.py
main_py_path = os.path.join(ROOT_DIR, "app", "main.py")
spec = importlib.util.spec_from_file_location("fastapi_main", main_py_path)
fastapi_main = importlib.util.module_from_spec(spec)
sys.modules["fastapi_main"] = fastapi_main
spec.loader.exec_module(fastapi_main)
fastapi_app = fastapi_main.app

# Custom full-screen wrapper displaying the exact localhost Dark Technical Dashboard
custom_css = """
body, html { 
    margin: 0 !important; 
    padding: 0 !important; 
    width: 100vw !important; 
    height: 100vh !important; 
    overflow: hidden !important; 
    background-color: #080c14 !important; 
}
.gradio-container { 
    padding: 0 !important; 
    margin: 0 !important; 
    max-width: 100vw !important; 
    height: 100vh !important; 
    background-color: #080c14 !important; 
}
footer { display: none !important; }
"""

html_iframe = """
<iframe 
    src="/static/index.html" 
    style="position:fixed; top:0; left:0; width:100vw; height:100vh; border:none; margin:0; padding:0; overflow-y:auto; z-index:999999;">
</iframe>
"""

with gr.Blocks(css=custom_css, title="VOICE RAG — HH Goa 2026") as demo:
    gr.HTML(html_iframe)

# Mount Gradio wrapper while serving the exact FastAPI backend & static assets
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    demo.launch()
