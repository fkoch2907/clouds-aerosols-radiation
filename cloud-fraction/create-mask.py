"""
Creates a mask for fish eye cloud camera images to remove objects at the horizont using the roipoly package.

usage: python create-mask.py, image used for mask creation and save name of mask need to be specified in script before running.
Output: npy file (can be converted to png using convert-mask-to-png.py)
"""

import matplotlib.pyplot as plt
from roipoly import RoiPoly
import numpy as np
from PIL import Image

data_path = "/Users/franziskakoch/Documents/Uni/UHH/second_semester/LEX/clouds-aerosols-radiation/Data/CloudCam/CAM4/WMD_CAM4_JPG_20260807/wmd_cam4_Image_20260807_171400_UTC.jpg"

# Load one representative image
img = np.array(Image.open(data_path))

# Show it and draw your polygon
masks = []
for name in ["trees", "roof1", "roof2"]:
    fig = plt.figure()
    plt.imshow(img)
    plt.title(f"Outline: {name}")
    roi = RoiPoly(color='r', fig=fig)
    plt.show()
    masks.append(roi.get_mask(img[:, :, 0]))

combined_mask = np.logical_or.reduce(masks)
circular_mask_img = np.array(Image.open("masks/MaskeHorizontGeomatikum.png"))
if circular_mask_img.ndim == 3:
    circular_mask_img = circular_mask_img[:, :, 0]

# Black (0) = outside the horizon circle = should be masked
circular_mask = circular_mask_img == 0
double_mask = np.logical_or(combined_mask, circular_mask)
np.save("maks_Westermarkelsdorf.npy", double_mask)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title("Original")
masked = img.copy()
masked[double_mask] = 0
ax[1].imshow(masked)
ax[1].set_title("Masked (obstructions blacked out)")
plt.show()
