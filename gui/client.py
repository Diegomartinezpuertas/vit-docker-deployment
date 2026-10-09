"""Call the Gradio GUI programmatically, as an API.

Usage (from the compose network):
    docker compose run --rm --no-deps -v ./samples:/samples gui \
        python client.py http://gui:7860 /samples/cat.jpg
"""
import sys

from gradio_client import Client, handle_file

target, image = sys.argv[1], sys.argv[2]
client = Client(target, verbose=False)
result = client.predict(handle_file(image), api_name="/predict")

print(f"Top-1: {result['label']}")
for c in result["confidences"]:
    print(f"  {c['label']:<30} {c['confidence']:.4f}")
