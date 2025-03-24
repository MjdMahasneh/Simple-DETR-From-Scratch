import os
import urllib.request
import tarfile

url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
save_path = "VOCtrainval_06-Nov-2007.tar"
extract_path = "./VOC2007"

# Download
if not os.path.exists(save_path):
    print("Downloading Pascal VOC 2007...")
    urllib.request.urlretrieve(url, save_path)
    print("Download done.")

# Extract
print("Extracting...")
with tarfile.open(save_path) as tar:
    tar.extractall(path=".")
print("Extraction complete. ✅")
