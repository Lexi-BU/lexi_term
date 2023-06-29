import wand as wand
import numpy as np

img_o = global_variables.data_lin["counts"]

# Rotate the image by 90 degrees
img_i = np.rot90(img_o)

with wand.image.Image.from_array(img_i) as img:
    img.save(filename="../figures/non_lin/original.png")


with wand.image.Image.from_array(img_i) as img:
    # Set background color to transparent
    img.background_color = wand.color.Color("transparent")
    # img.background_color = img[0, 0]
    # If the background color is not set, then the image will be black
    # img.alpha_channel = "remove"

    args = (0.1, 0.1, 0.1, 1.3)
    img.distort("barrel_inverse", args)

    # Save the image to a pdf file and set its resolution to 300 dpi
    img.save(filename="../figures/non_lin/barrel_inverse.png")


# Apply the barrel distortion correction to the image

with wand.image.Image.from_array(img_i) as img:
    img.background_color = wand.color.Color("transparent")
    # img.background_color = img[0, 0]
    # If the background color is not set, then the image will be black
    # img.alpha_channel = "remove"

    img.virtual_pixel = "background"

    args = (0.0, 0.1, -0.1, 1.1)
    img.distort("barrel", args)

    # Save the image to a pdf file and set its resolution to 300 dpi
    img.save(filename="../figures/non_lin/barrel.png")


def barrel_correction(r, k1, k2, k3, p1, p2):
    """
    Barrel distortion correction function

    Parameters
    ----------
    r : numpy.ndarray
        The radial distance from the center of the image
    k1 : float
        The first radial distortion coefficient
    k2 : float
        The second radial distortion coefficient
    k3 : float
        The third radial distortion coefficient
    p1 : float
        The first tangential distortion coefficient
    p2 : float
        The second tangential distortion coefficient

    Returns
    -------
    numpy.ndarray
        The corrected radial distance from the center of the image
    """
    r2 = r**2
    r4 = r2**2
    r6 = r4 * r2
    r8 = r6 * r2
    return r * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p1 * r * r2 + p2 * (r4 + 2 * r2)


def barrel_correction_inverse(r, k1, k2, k3, p1, p2):
    """
    Barrel distortion correction function

    Parameters
    ----------
    r : numpy.ndarray
        The radial distance from the center of the image
    k1 : float
        The first radial distortion coefficient
    k2 : float
        The second radial distortion coefficient
    k3 : float
        The third radial distortion coefficient
    p1 : float
        The first tangential distortion coefficient
    p2 : float
        The second tangential distortion coefficient

    Returns
    -------
    numpy.ndarray
        The corrected radial distance from the center of the image
    """
    r2 = r**2
    r4 = r2**2
    r6 = r4 * r2
    r8 = r6 * r2
    return r / (1 + k1 * r2 + k2 * r4 + k3 * r6) - 2 * p1 * r * r2 - p2 * (r4 + 2 * r2)


img = global_variables.data_lin["counts"]

# img = img.T
# # Convert the ndarray to matrix
# # img = np.matrix(img)
#
# # Apply the barrel distortion correction to the image
# img = barrel_correction(img, 0.0, 0.1, -0.1, 0.1, 0.1)
#
# # Save the array as an image
# with wand.image.Image.from_array(img) as img:
#     img.save(filename="../figures/test_nln_v2.png")
