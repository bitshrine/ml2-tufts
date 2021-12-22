import os
import numpy as np
import pandas as pd

from basic import rescale_image
from basic import separate

def load_df(name):
    """
    Returns the dataframe resulting from
    the processing of an image. Attempts
    to read it from a file, but generates
    it if it does not yet exist.
    """
    src = '../output/{name}.json'.format(name=name)

    if (not os.path.exists(src)):
        print("File has not yet been processed.")
        os.system(
            'python3 img_processing.py --source=../data/{name}.tiff'.format(name=name))

    return pd.read_json(src)


def mid_generator(width):
    """
    Generator function which starts with the middle
    index and then progressively yields indices farther
    from the middle.
    """
    mid = int(np.floor(width / 2))
    lo = list(range(mid))
    lo.reverse()
    hi = list(range(mid, width))

    i = 0
    while (lo or hi):
        i = (i + 1) % 2
        if (i == 0):
            y = lo[0]
            lo = lo[1:]
        else:
            y = hi[0]
            hi = hi[1:]

        yield y


def get_lower_brightest_pixel(img, bg_threshold=0.2):
    """
    Return the coordinates of the brightest
    pixel closest to the middle of the bottom
    of the image. The `bg_threshold` parameter
    indicates the value above which we consider
    pixels to not be part of the background.
    """

    for i in range(img.shape[0] - 1).__reversed__():
        width = len(img[i])
        for j in mid_generator(width):
            if (img[i, j] > bg_threshold):
                return (i, j)

    return None

from blobs import coordinates_within_bounds

def pixel_is_similar(img, cur_, next_, sim_delta=0.025):
    """
    Check that the next pixel is within image
    boundaries and has a value similar enough to
    the current pixel.
    """
    return coordinates_within_bounds(img, cur_, next_)\
        and (np.abs(img[cur_] - img[next_]) < sim_delta)


from blobs import get_blob_pixels, index_set

from scipy import ndimage
from scipy.ndimage.filters import maximum_filter, minimum_filter


def find_closest_point(pt, idx):
    c_x, c_y = pt

    min_dist = np.inf
    min_idx = -1
    for i in range(len(idx[0])):
        x, y = (idx[0][i], idx[1][i])
        dist = np.sqrt(((x - c_x) ** 2) + ((y - c_y) ** 2))
        if (dist < min_dist):
            min_dist = dist
            min_idx = i

    return (idx[0][min_idx], idx[1][min_idx])


def naive_line(r0, c0, r1, c1):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = naive_line(c0, r0, c1, r1)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return naive_line(r1, c1, r0, c0)

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * (r1-r0) / (c1-c0) + (c1*r0-c0*r1) / (c1-c0)

    valbot = np.floor(y)-y+1
    valtop = y-np.floor(y)

    return (np.concatenate((np.floor(y), np.floor(y)+1)).astype(int), np.concatenate((x, x)).astype(int),
            np.concatenate((valbot, valtop)))



def get_tuft_blob(img):
    or_img = img.copy()[1:-1, 1:-1]
    img = rescale_image(img[1:-1, 1:-1])

    img = ndimage.filters.gaussian_filter(img, 1.25)

    footprint = int(np.floor(img.shape[0]/2))

    maxima = (img == maximum_filter(img, footprint)).astype(np.float16)
    max_idx = np.where(maxima == 1.0)

    minima = (img == minimum_filter(img, footprint)).astype(np.float16)
    min_idx = np.where(minima == 1.0)
    #min_idx = (min_idx[0][0], min_idx[1][0])

    closest = find_closest_point((img.shape[0]/2, img.shape[1]/2), max_idx)

    #bottom_mid = (img.shape[0] - 1, int(np.floor(img.shape[1] / 2)))


    #line_img = np.zeros(img.shape)
    #xx, yy, val = naive_line(
    #    closest[0], closest[1], bottom_mid[0], bottom_mid[1])

    #l = len(xx)
    #xx, yy = xx[:l], yy[:l]
    #xx -= 1
    #yy -= 1

    #line_img[xx, yy] = 1

    #brighten the area
    #img[np.where(line_img == 1)] = np.max(val)  # 1#img + line_img * 0.05
    #img = rescale_image(img)

    inv_img = 1 - img
    inv_img = (inv_img * (10 ** 3)).astype(np.uint16)

    mask = np.zeros(shape=inv_img.shape, dtype=bool)
    mask[max_idx] = True
    mask[min_idx] = True

    markers, _ = ndimage.label(mask)

    labels = ndimage.watershed_ift(inv_img, markers)

    result = labels.copy()

    zones = list(np.unique(markers))

    zones.remove(markers[closest])

    for l in zones:
        result[np.where(labels == l)] = 0

    result[np.where(labels == markers[closest])] = 1

    frame = (50, 20)

    crop = np.zeros(shape=frame)
    crop[-img.shape[0]:frame[0], -img.shape[1]:frame[1]] = result
    result = crop

    return result


def tuft_processor(img):

    

    pass
    #result_img = img.copy()

    #result_img = separate(result_img, 10)

    #return get_tuft_blob(result_img)

    
