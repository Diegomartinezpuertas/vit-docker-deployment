# The GUI only sends HTTP requests: no PyTorch, no CUDA. A slim Python image
# keeps it a few hundred MB instead of the ~12 GB inference image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_URL=http://inference:8000

# Run as a non-root user.
RUN useradd --create-home --uid 1000 user

WORKDIR /app

# Copy ONLY the requirements first so this slow layer is cached until they change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code LAST: it changes most often.
COPY --chown=user:user . .

USER user

EXPOSE 7860

# Uses Python's urllib instead of curl, so no apt-get layer is needed.
# urlopen raises on connection errors and HTTP 4xx/5xx -> exit code 1.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=4)" || exit 1

CMD ["python", "app.py"]
