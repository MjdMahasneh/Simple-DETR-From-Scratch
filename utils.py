import math
import torch

# --- Collate Function ---
def collate_fn(batch):
    return {
        "images": torch.stack([item["images"] for item in batch]),
        "labels": [item["labels"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
    }

# --- Learning Rate Scheduler ---
def lr_lambda(epoch):
    if epoch < 10:  # warm-up (factor of base lr) for 10 epochs
        return epoch / 10
    else:
        return 0.5 * (1 + math.cos((epoch - 10) / (T_max - 10) * math.pi))