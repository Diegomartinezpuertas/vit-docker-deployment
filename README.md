# ViT inference in Docker

Image classification with a Vision Transformer (`google/vit-base-patch16-224`),
split into two containers with Docker Compose:

- `inference/`: FastAPI + PyTorch on the GPU (also used to fine-tune the model).
- `gui/`: Gradio web interface that calls the API.
- `docs/`: report (PDF), figures and logs.

Requirements: Docker with Compose v2 and the NVIDIA Container Toolkit.

## Run

```bash
docker compose up -d --build        # API + GUI
docker compose ps                   # both should be (healthy)
```

GUI: http://localhost:7861 (port set in `.env`). The API is only reachable
inside the Compose network.

Terminal inference against the API container:

```bash
docker compose exec inference curl -s localhost:8000/health
docker compose cp samples/cat.jpg inference:/tmp/cat.jpg
docker compose exec inference curl -s -F "file=@/tmp/cat.jpg" localhost:8000/predict
```

Through the GUI's API (gradio_client):

```bash
docker compose run --rm --no-deps -v ./samples:/samples gui \
    python client.py http://gui:7860 /samples/cat.jpg
```

## Fine-tune on the GPU

```bash
docker compose --profile train run --rm train
```

Then set `MODEL_ID=/models/vit-beans` in `.env` and `docker compose up -d`.

## Stop

```bash
docker compose down        # add -v to also delete the model volume
```
