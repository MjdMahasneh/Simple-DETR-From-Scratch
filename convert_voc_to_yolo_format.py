import os
import xml.etree.ElementTree as ET

voc_root = r"F:\datasets\New folder\VOCdevkit\VOC2007"
voc_anno = os.path.join(voc_root, "Annotations")
voc_images = os.path.join(voc_root, "JPEGImages")
yolo_labels = os.path.join(voc_root, "labels")

os.makedirs(yolo_labels, exist_ok=True)

# Class list from VOC
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

for xml_file in os.listdir(voc_anno):
    tree = ET.parse(os.path.join(voc_anno, xml_file))
    root = tree.getroot()

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in VOC_CLASSES:
            continue
        cls_id = VOC_CLASSES.index(cls_name)

        bbox = obj.find("bndbox")
        x_min = float(bbox.find("xmin").text)
        y_min = float(bbox.find("ymin").text)
        x_max = float(bbox.find("xmax").text)
        y_max = float(bbox.find("ymax").text)

        # Convert to YOLO format
        cx = ((x_min + x_max) / 2) / img_w
        cy = ((y_min + y_max) / 2) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # Write .txt
    img_id = os.path.splitext(xml_file)[0]
    with open(os.path.join(yolo_labels, f"{img_id}.txt"), "w") as f:
        f.write("\n".join(yolo_lines))
