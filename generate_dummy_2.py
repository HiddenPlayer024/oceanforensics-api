import numpy as np
from PIL import Image

img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
Image.fromarray(img).save('/home/unknown/Projects/oil_spill_api/dataset/images/dummy2.png')

mask = np.zeros((256, 256), dtype=np.uint8)
mask[50:100, 50:80] = 255
Image.fromarray(mask).save('/home/unknown/Projects/oil_spill_api/dataset/masks/dummy2.png')
