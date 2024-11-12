import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import math
from skimage.transform import rotate, matrix_transform
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

# We open im.show to find coordinates for where we can place landmarks on the destination
# But it's easier to just read from gimp ;)
plt.imshow(dst_img)
plt.title("Hover over the image to see coordinates")
plt.colorbar()  # Optional: adds a colorbar for pixel values if needed
plt.show()

# We insert the land marks at
dst = np.array([[620, 292], [381, 161], [196, 273], [278, 441], [598, 447]])



# We plot landmarks and see if its good
fig, ax = plt.subplots()
ax.plot(src[:, 0], src[:, 1], '-r', markersize=12, label="Source")
ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
ax.invert_yaxis()
ax.legend()
ax.set_title("Landmarks before alignment")
plt.show()

""" We use objective function to see how well the landmarks allign"""
e_x = src[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f = error_x + error_y
print(f"Landmark alignment error F: {f}")

"""We find the optimal euclidian tranformation to allign with"""
tform = EuclideanTransform()
tform.estimate(src, dst)
# We apply it to the source points
src_transform = matrix_transform(src, tform.params)

# We look at the new allignment
fig, ax = plt.subplots()
ax.plot(src_transform[:, 0], src_transform[:, 1], '-r', markersize=12, label="Source")
ax.plot(dst[:, 0], dst[:, 1], '-g', markersize=12, label="Destination")
ax.invert_yaxis()
ax.legend()
ax.set_title("Landmarks after alignment")
plt.show()

#We check the objective function again

e_x = src_transform[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src_transform[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f = error_x + error_y
print(f"Landmark alignment error after allignment F: {f}")

"""We apply the transformation to the image itself"""
warped = warp(src_img, tform.inverse)
blend = 0.5 * img_as_float(warped) + 0.5 * img_as_float(dst_img)
io.imshow(blend)
io.show()