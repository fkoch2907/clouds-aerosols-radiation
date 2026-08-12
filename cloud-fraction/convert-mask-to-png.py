"""

Convert a numpy mask to a PNG image.

Usage: python convert-mask-to-png.py, specify filename of mask and save name before running
Output: png that is black everywhere where the mask is and transparent everywhere else

"""

import numpy as np
import matplotlib.pyplot as plt

mask = np.load("masks/mask_Burg.npy").astype(bool)  # True = masked area

rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
rgba[mask] = [0, 0, 0, 255]   # black, opaque
rgba[~mask] = [0, 0, 0, 0]    # transparent

plt.imsave("masks/mask_Burg.png", rgba)