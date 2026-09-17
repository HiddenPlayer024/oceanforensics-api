import os
import shutil
import numpy as np
from PIL import Image

src_class1 = 'dataset/kaggle/data/Class_1'
src_class0 = 'dataset/kaggle/data/Class_0'

img_dest = 'dataset/train/images'
mask_dest = 'dataset/train/masks'

os.makedirs(img_dest, exist_ok=True)
os.makedirs(mask_dest, exist_ok=True)

# Function to copy images and generate masks
def process_class(src_dir, is_oil):
    files = os.listdir(src_dir)
    for f in files:
        if not f.endswith('.jpg'): continue
        
        src_path = os.path.join(src_dir, f)
        dest_path = os.path.join(img_dest, f)
        shutil.copy(src_path, dest_path)
        
        # generate mask
        img = Image.open(src_path)
        w, h = img.size
        
        if is_oil:
            # We'll create a centered blob as a mask for 'Class_1' since we don't have true polygons
            # To train a segmentation model properly, a center blob is better than a full image.
            # But full 255s is also fine. Let's just do full 255s.
            mask_arr = np.ones((h, w), dtype=np.uint8) * 255
        else:
            mask_arr = np.zeros((h, w), dtype=np.uint8)
            
        mask_img = Image.fromarray(mask_arr)
        mask_img.save(os.path.join(mask_dest, f))

process_class(src_class1, True)
process_class(src_class0, False)

print("Dataset prepared.")
