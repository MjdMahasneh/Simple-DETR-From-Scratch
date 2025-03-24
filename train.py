import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from loss import HungarianMatcher, SetCriterion
from datasets import YOLODetectionDataset
from detr import DETR
from utils import collate_fn, lr_lambda


# --- Config ---
NUM_EPOCHS = 500
T_max = 500
NUM_CLASSES = 20
LOAD_PRETRAINED = True
PRETRAINED_PATH = "./ckpt/detr_best.pth"
IMAGE_DIR = "./VOC2007/VOCdevkit/VOC2007/JPEGImages"
LABEL_DIR = "./VOC2007/VOCdevkit/VOC2007/labels"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 32
NUM_WORKERS = 4
CKPT_PATH = "./ckpt/detr_best.pth" # Save path for best model




if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset & DataLoader ---
    dataset = YOLODetectionDataset(
        image_dir= IMAGE_DIR,
        label_dir = LABEL_DIR,
        transform=T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomAutocontrast(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.ToTensor(),
        ])
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model Setup ---
    model = DETR(num_classes=NUM_CLASSES).to(device)

    ## Load pretrained weights
    if LOAD_PRETRAINED:
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location="cpu"))

    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=NUM_CLASSES, matcher=matcher).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda)


    # --- Training Loop ---
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = batch["images"].to(device)
            targets = [{"labels": lbl.to(device), "boxes": box.to(device)}
                       for lbl, box in zip(batch["labels"], batch["boxes"])]

            outputs = model(images)
            indices = matcher(outputs, targets)
            loss_cls = criterion.loss_labels(outputs, targets, indices)
            loss_bbox = criterion.loss_boxes(outputs, targets, indices)
            loss = loss_cls + loss_bbox

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_bbox_loss += loss_bbox.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_bbox_loss = total_bbox_loss / len(dataloader)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Cls: {avg_cls_loss:.4f} | Box: {avg_bbox_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("./ckpt", exist_ok=True)
            torch.save(model.state_dict(), CKPT_PATH)
            print("✅ Best model saved.")
