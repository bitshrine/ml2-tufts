import os
import numpy as np
import pandas as pd

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

def pixel_is_similar(img, cur_, next_, sim_delta=0.15):
    """
    Check that the next pixel is within image
    boundaries and has a value similar enough to
    the current pixel.
    """
    return coordinates_within_bounds(img, cur_, next_)\
        and (np.abs(img[cur_] - img[next_]) < sim_delta)


from blobs import get_blob_pixels, index_set

def tuft_pipeline(img):
    start_pixel = get_lower_brightest_pixel(img, 0.2)

    main_blob = get_blob_pixels(img, start_pixel, pixel_is_similar, set())

    idx = index_set(main_blob)

    result_img = np.zeros(shape=img.shape)
    result_img[idx] = 1

    return result_img
