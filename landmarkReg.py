import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import math
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage.util import img_as_float

"""Landmark based registration
The art of aligning a source image to fit a destination image"""

#Data
src_img_path = "data/Hand1.jpg"
dst_img_path = "data/Hand2.jpg"

src_img = io.imread(src_img_path)
dst_img = io.imread(dst_img_path)

# How they fit eachother at the start
blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
io.imshow(blend)
io.show()

# See manual landmarks
src = np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]])
plt.imshow(src_img)
plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
plt.show()