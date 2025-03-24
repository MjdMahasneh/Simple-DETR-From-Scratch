import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Resize, Compose
import os
from loss import box_cxcywh_to_xyxy
from detr import DETR

VOC_CLASSES = {
    0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
    5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
    10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
    15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
}

# --- Load model ---
NUM_CLASSES = 20
model = DETR(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("./ckpt/detr_best.pth", map_location="cpu"))
model.eval()

image_size = (512, 512)
# --- Image transform ---
transform = Compose([
    Resize(image_size),
    ToTensor()
])

image_dir = "F:/datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages"

sample_image_name = os.listdir(image_dir)[20]  # grab first image
print(sample_image_name)
image_path = os.path.join(image_dir, sample_image_name)

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]

# --- Inference ---
with torch.no_grad():
    outputs = model(input_tensor)

probs = outputs["pred_logits"].softmax(-1)[0]  # [num_queries, num_classes+1]
# boxes = outputs["pred_boxes"][0]               # [num_queries, 4]
#todo test with clamp
boxes = outputs["pred_boxes"][0].clamp(0, 1)  # Ensure in bounds

print(probs.shape, boxes.shape)
print('probs:', probs)
print('boxes:', boxes)

# --- Filter predictions ---
conf_thresh = 0.90

scores, labels = probs.max(-1)
keep = (scores > conf_thresh) & (labels != NUM_CLASSES)
boxes = box_cxcywh_to_xyxy(boxes[keep]) * 512
boxes = boxes.clamp(0, image_size[0])  # Clamp to image bounds (0-512)

# --- Check for out-of-bound boxes ---
invalid_boxes = []
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    if not (0 <= x1 <= 512 and 0 <= y1 <= 512 and 0 <= x2 <= 512 and 0 <= y2 <= 512):
        invalid_boxes.append((i, box, scores[i].item(), labels[i].item()))

if invalid_boxes:
    print("❗ Out-of-bound boxes detected:")
    for idx, box, score, label in invalid_boxes:
        print(f"  idx {idx}: {VOC_CLASSES.get(label, 'no-object')} | score: {score:.2f} | box: {box}")

scores = scores[keep]
labels = labels[keep]

# --- Draw ---
draw = image.copy()
plt.figure(figsize=(8, 8))
plt.imshow(draw)
ax = plt.gca()

for box, score, label in zip(boxes, scores, labels):
    class_name = VOC_CLASSES.get(label.item(), "unknown")
    x1, y1, x2, y2 = box
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               edgecolor='lime', facecolor='none', lw=2))
    ax.text(x1, y1 - 4, f"{class_name} {score:.2f}", color="white",
            bbox=dict(facecolor='green', alpha=0.5), fontsize=10)

plt.axis("off")
plt.title("Predictions")
plt.show()
