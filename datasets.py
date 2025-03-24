import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class YOLODetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform or T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, image_name)
        lbl_path = os.path.join(self.label_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        boxes = []
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    try:
                        class_id, cx, cy, w, h = map(float, line.strip().split())
                        boxes.append([cx, cy, w, h])
                        labels.append(int(class_id))
                    except Exception as e:
                        # print(f"Error reading {lbl_path}: {e}")
                        pass

        # if len(boxes) == 0:
        #     print(f"⚠️ Empty label: {lbl_path}")

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return {"images": image, "labels": target["labels"], "boxes": target["boxes"]}