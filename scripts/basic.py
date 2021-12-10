import numpy as np

def rescale_image(img):
    """
    Rescale the image's grayscale values to
    the 0-1 range.
    """
    result = img.copy()
    minval, maxval = np.min(img), np.max(img)
    if (maxval != minval):
        result = (result - minval)/(maxval - minval)
    return result


def separate(img, factor):
    """
    Separate the images by making bright pixels
    brighter and dim pixels dimmer
    """
    result = img.copy()
    result = ((img + 0.1)*(img - 1.1)) ** factor
    return np.abs(rescale_image(result))


def threshold(img, alpha):
    """
    Make all pixels with values above of equal
    to `alpha` be 1, and all others be 0.
    """
    result = img.copy()
    idx = result >= alpha
    result[idx] = 1
    result[~idx] = 0

    return result
