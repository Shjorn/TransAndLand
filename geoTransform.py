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

"""A function to compare two pictures"""
def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed)
    ax2.set_title(transformed_name)
    ax2.axis('off')
    io.show()

#Data
im_org_path = "data/NusaPenida.png"
im_org = io.imread(im_org_path)

"""Rotating an image"""

# angle in degrees - counter clockwise from middle point
rotation_angle = 10
rotated_img = rotate(im_org, rotation_angle)
show_comparison(im_org, rotated_img, "Rotated image")

# angle in degrees - counter clockwise from specified point
rot_center = [0, 0]
rotated_img = rotate(im_org, rotation_angle, center=rot_center)
show_comparison(im_org, rotated_img, "Rotated image, top-left")

# angle in degrees - counter clockwise from middle point, reflected background (warp is also an option)
rotated_img = rotate(im_org, rotation_angle, mode="reflect")
show_comparison(im_org, rotated_img, "Rotated image, ref")

# angle in degrees - counter clockwise from middle point, seamless rotation
rotated_img = rotate(im_org, rotation_angle, resize=True)
show_comparison(im_org, rotated_img, "Rotated image, full")

# angle in degrees - counter clockwise from middle point, seamless rotation and solid background (0 to 1)
rotated_img = rotate(im_org, rotation_angle, resize=True, mode="constant", cval=1)
show_comparison(im_org, rotated_img, "Rotated image, nice full")


"""Euclidian or rigid body transformation. Using rotation and translation"""

# angle in radians - counter clockwise
# the transformation consists of a 3 x 3 matrix. The matrix is used to transform points using homogenous coordinates
rotation_angle = 10.0 * math.pi / 180.
trans = [10, 20]
tform = EuclideanTransform(rotation=rotation_angle, translation=trans)
print(tform.params)

# Using warp we apply the transformation to the image
# Note: The warp function actually does an inverse transformation of the image,
# since it uses the transform to find the pixels values in the input image that should be placed in the output image.
transformed_img = warp(im_org, tform)
show_comparison(im_org, transformed_img, "Euclidian image")

# Getting the inverse of our transformation
transformed_img = warp(im_org, tform.inverse)
show_comparison(im_org, transformed_img, "Inverse euclidian image")


"""Similarity transform. Using translation, rotation and a scaling"""
# angle in radians - counter clockwise
rotation_angle = 15.0 * math.pi / 180.
trans = [40, 30]
scale = 0.6

# Create a SimilarityTransform
tform = SimilarityTransform(scale=scale, rotation=rotation_angle, translation=trans)
print(tform.params)


# Using warp we apply the transformation to the image
# Note: The warp function actually does an inverse transformation of the image,
# since it uses the transform to find the pixels values in the input image that should be placed in the output image.
transformed_img = warp(im_org, tform)
show_comparison(im_org, transformed_img, "similarity image")


"""Swirl image, does as the name implies"""
str = 10
rad = 300
swirl_img = swirl(im_org, strength=str, radius=rad)
show_comparison(im_org, swirl_img, "Swirled image")

#We can also specify the location of the swirl
str = 10
rad = 300
c = [500, 400]
swirl_img = swirl(im_org, strength=str, radius=rad, center=c)
show_comparison(im_org, swirl_img, "Swirled image")


