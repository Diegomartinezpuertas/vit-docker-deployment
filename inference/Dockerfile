# Official PyTorch 2.7 + CUDA 12.8 image (CUDA 12.8+ is required for Blackwell GPUs like the RTX 5070)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

# curl is needed by the HEALTHCHECK. Update, install and clean up in ONE layer:
# deleting the apt lists in a later RUN would not shrink the image.
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Non-root user. /models is created here owned by that user, so a new named
# volume mounted on it inherits the ownership and the user can write weights.
RUN useradd --create-home --uid 1000 user && mkdir /models && chown user:user /models

WORKDIR /app

# Copy ONLY the requirements first so this slow layer is cached until they change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code LAST: it changes most often, so editing app.py
# only invalidates this layer and the pip install above stays cached.
COPY --chown=user:user . .

USER user

EXPOSE 8000

# -f makes curl fail on HTTP errors (4xx/5xx); start-period gives the model time to load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
