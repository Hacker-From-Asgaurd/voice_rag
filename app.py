import os
import uvicorn
import gradio as gr
from app.main import app as fastapi_app

# Mount Gradio Blocks for Hugging Face SDK detection while serving your custom FastAPI Dark UI
with gr.Blocks() as demo:
    pass

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
