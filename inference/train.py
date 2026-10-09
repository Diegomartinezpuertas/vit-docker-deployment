"""Fine-tune ViT on the `beans` dataset (3 leaf classes) and save it for serving.

Runs in the same image as the API (compose service `train`). The result is
written to the shared volume, so the API serves it by setting MODEL_ID.
"""
import argparse
import json
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", default="google/vit-base-patch16-224")
parser.add_argument("--output", default="/models/vit-beans")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

ds = load_dataset("AI-Lab-Makerere/beans")
labels = ds["train"].features["labels"].names
id2label = dict(enumerate(labels))
label2id = {name: i for i, name in id2label.items()}

processor = AutoImageProcessor.from_pretrained(args.base_model, use_fast=True)
# Replace the 1000-class ImageNet head with a new 3-class head.
model = AutoModelForImageClassification.from_pretrained(
    args.base_model,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
).to(device)


def transform(batch):
    # Applied lazily when the DataLoader workers read a batch.
    images = [img.convert("RGB") for img in batch["image"]]
    out = processor(images=images, return_tensors="pt")
    out["labels"] = batch["labels"]
    return out


def collate(rows):
    return {
        "pixel_values": torch.stack([r["pixel_values"] for r in rows]),
        "labels": torch.tensor([r["labels"] for r in rows]),
    }


def loader(split, shuffle):
    # num_workers > 0 uses separate processes that pass tensors through
    # /dev/shm: this is why the container needs --shm-size / shm_size.
    return DataLoader(ds[split].with_transform(transform), batch_size=args.batch_size,
                      shuffle=shuffle, num_workers=args.workers, collate_fn=collate)


def evaluate(split):
    model.eval()
    correct = total = 0
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for batch in loader(split, shuffle=False):
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            correct += (logits.argmax(-1).cpu() == batch["labels"]).sum().item()
            total += len(batch["labels"])
    return correct / total


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
history = []
print(f"Baseline val accuracy (random head): {evaluate('validation'):.3f}")

for epoch in range(1, args.epochs + 1):
    model.train()
    start, running_loss, steps = time.time(), 0.0, 0
    for batch in loader("train", shuffle=True):
        optimizer.zero_grad()
        # bf16 mixed precision: faster on the GPU's tensor cores, less VRAM.
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = model(pixel_values=batch["pixel_values"].to(device),
                         labels=batch["labels"].to(device)).loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        steps += 1
    val_acc = evaluate("validation")
    row = {"epoch": epoch, "train_loss": round(running_loss / steps, 4),
           "val_accuracy": round(val_acc, 4), "seconds": round(time.time() - start, 1)}
    history.append(row)
    print(row)

test_acc = evaluate("test")
print(f"Test accuracy: {test_acc:.3f}")
if device == "cuda":
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

model.save_pretrained(args.output)
processor.save_pretrained(args.output)
with open(f"{args.output}/metrics.json", "w") as f:
    json.dump({"history": history, "test_accuracy": round(test_acc, 4)}, f, indent=2)
print(f"Saved to {args.output}")
