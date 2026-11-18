import sys
import os
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Server environment configuration for Hugging Face Spaces
os.environ["HOST"] = "0.0.0.0"
os.environ["PORT"] = "7860"
os.environ["CORS_ORIGINS"] = "*"

import uvicorn
import gradio as gr
from backend.main import app as fastapi_app
from backend.services.classifier import DistilBERTClassifier

# Simple web interface shown when viewing the Hugging Face Space directly in a browser
with gr.Blocks(title="TruthLens AI Backend") as demo:
    gr.Markdown("# 🔍 TruthLens AI — Backend API")
    gr.Markdown(
        """
        **Status:** 🟢 **Online & Operational**  
        This Space hosts the **TruthLens AI** backend powering your Vercel frontend.
        
        - **Swagger Docs:** [Open /docs](/docs)
        - **Health Check:** [Open /health](/health)
        """
    )
    with gr.Row():
        test_text = gr.Textbox(
            label="Quick Test Claim / Headline",
            placeholder="NASA detects water on exoplanet...",
            lines=2
        )
    test_btn = gr.Button("Classify Style (DistilBERT)")
    output_json = gr.JSON(label="Result")

    def test_classify(text):
        if not text:
            return {"error": "Please enter input text."}
        clf = DistilBERTClassifier.get_instance()
        return clf.predict(text).model_dump()

    test_btn.click(fn=test_classify, inputs=test_text, outputs=output_json)

# Mount Gradio onto the existing FastAPI application
# All FastAPI endpoints (/api/verify, /api/classify, /health, /docs) remain available at the root!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
