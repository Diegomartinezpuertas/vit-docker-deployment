import os

import gradio as gr
import requests

# Base URL of the inference API. In Compose this is the service name
# (http://inference:8000); "localhost" would point to THIS container.
API_URL = os.getenv("API_URL", "http://localhost:8000")


def classify(image_path):
    # Gradio gives us the uploaded image as a temporary file path.
    if image_path is None:
        raise gr.Error("Upload an image first")

    # Send the raw file to the API as multipart/form-data, field name "file"
    # (must match the parameter name of the FastAPI /predict endpoint).
    try:
        with open(image_path, "rb") as f:
            response = requests.post(f"{API_URL}/predict", files={"file": f}, timeout=60)
    except requests.ConnectionError:
        raise gr.Error(f"Cannot reach the inference API at {API_URL}")

    if response.status_code != 200:
        raise gr.Error(f"API error {response.status_code}: {response.text}")

    # gr.Label expects a dict {label: confidence}.
    predictions = response.json()["predictions"]
    return {p["label"]: p["score"] for p in predictions}


def api_info():
    # Ask the API which model/device it serves, to show it in the page header.
    try:
        info = requests.get(f"{API_URL}/health", timeout=5).json()
        return f"Model `{info['model']}` running on `{info['device']}` in a separate FastAPI container."
    except requests.RequestException:
        return "Inference API not reachable yet."


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="filepath", label="Image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="ViT image classifier",
    description=api_info(),
    flagging_mode="never",
    # Stable endpoint name for programmatic clients (gradio_client, see client.py).
    api_name="predict",
)

if __name__ == "__main__":
    # 0.0.0.0 = listen on all interfaces; 127.0.0.1 would be unreachable
    # from outside the container even with -p 7860:7860.
    demo.launch(server_name="0.0.0.0", server_port=7860)
