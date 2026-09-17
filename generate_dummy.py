import os
import numpy as np
from PIL import Image

os.makedirs('/home/unknown/Projects/oil_spill_api/dataset/images', exist_ok=True)
os.makedirs('/home/unknown/Projects/oil_spill_api/dataset/masks', exist_ok=True)

# Generate a noisy synthetic SAR image
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
Image.fromarray(img).save('/home/unknown/Projects/oil_spill_api/dataset/images/dummy.png')

# Generate a synthetic mask
mask = np.zeros((256, 256), dtype=np.uint8)
mask[100:150, 100:130] = 255
Image.fromarray(mask).save('/home/unknown/Projects/oil_spill_api/dataset/masks/dummy.png')
