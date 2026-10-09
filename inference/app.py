import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Model to serve. Read from an env var so we can switch models (e.g. our
# fine-tuned one in /models/vit-beans) without rebuilding the image.
MODEL_ID = os.getenv("MODEL_ID", "google/vit-base-patch16-224")

# Use the GPU if the container can see one (--gpus / deploy.devices),
# otherwise fall back to CPU so the same image runs anywhere.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Holds the processor and model. Filled once at startup, shared by all requests.
ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: loading the model is slow (seconds),
    # so we never want to do it inside a request.
    ml["processor"] = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    # eval() disables training-only layers like dropout -> deterministic output.
    ml["model"] = model.to(DEVICE).eval()
    yield
    # Runs once on shutdown: free the model memory.
    ml.clear()


app = FastAPI(title="ViT inference", lifespan=lifespan)


@app.get("/health")
def health():
    # Used by the Docker HEALTHCHECK and by compose's depends_on condition.
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID}


# Plain `def` (not `async def`): inference is CPU/GPU-bound and would block the
# event loop, so FastAPI runs sync endpoints in a thread pool instead.
@app.post("/predict")
def predict(file: UploadFile = File(...), top_k: int = 5):
    processor = ml["processor"]
    model = ml["model"]

    # 1. Decode the uploaded bytes into an RGB image (PNGs may be RGBA,
    #    some JPEGs are grayscale; the model expects 3 channels).
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    # 2. Preprocess: resize to 224x224, normalize, convert to a tensor of
    #    shape [1, 3, 224, 224]. Then move every tensor to the model's device.
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}

    # 3. Forward pass. inference_mode() turns off gradient tracking:
    #    less memory and faster, since we are not training.
    with torch.inference_mode():
        logits = model(**inputs).logits  # shape [1, num_classes], raw scores

    # 4. Turn raw scores into probabilities and keep the k most likely classes.
    probs = logits.softmax(dim=-1)[0]  # [0] drops the batch dimension
    top_k = max(1, min(top_k, probs.shape[0]))
    scores, indices = torch.topk(probs, top_k)

    # 5. Map class indices to human-readable labels. .item() converts a
    #    one-element tensor into a plain Python number that JSON can encode.
    predictions = [
        {"label": model.config.id2label[idx.item()], "score": round(score.item(), 4)}
        for score, idx in zip(scores, indices)
    ]
    return {"predictions": predictions}
