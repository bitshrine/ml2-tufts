import numpy as np
import pandas as pd


def create_blobs(img, radius, ratio_threshold):
    """
    Create blobs on the image using a marching squares
    method. If the ratio of the number of white pixels
    around any pixel to the total number of surrounding
    pixels if above or equal to `ratio_threshold`, the
    pixel is set to 1, and 0 otherwise.
    """
    result = img.copy()
    total = ((2*radius + 1) ** 2) - 1
    for x in range(radius, len(img) - radius):
        for y in range(radius, len(img[0]) - radius):
            center = 1 if img[x, y] == 1 else 0
            n_white = np.count_nonzero(
                img[x-radius:x+radius, y-radius:y+radius]) - center
            r = n_white/total
            result[x, y] = 1 if r >= ratio_threshold else 0

    return result


def always_true_validator(img, cur_, next_):
    """
    Example function which performs
    an operation on two pixels from an image
    and returns a boolean value indicating
    if the pixel indexed by the `next_` parameter
    is valid
    """
    return True


def coordinates_within_bounds(img, next_):
    """
    Checks whether the given coordinates are inside the
    image boundaries.
    """
    x, y = next_
    return (x >= 0 and x < len(img) and y >= 0 and y < len(img[0]))


coord_comb = [(-1, -1), (-1,  0), (-1,  1),
              (0, -1),           (0,  1),
              (1, -1), (1,  0), (1,  1)]


def get_blob_pixels(img, coords, pixel_validator=always_true_validator, pixel_set=set()):
    """Returns a set of pixel coordinates which belong to a blob"""
    x, y = coords
    if (img[x, y] == 0) or ((x, y) in pixel_set):
        return set()
    else:
        pixel_set.add((x, y))
        for i, j in coord_comb:
            if (coordinates_within_bounds(img, (x+i, y+j)) and pixel_validator(img, (x, y), (x+i, y+j))):
                pixel_set.update(get_blob_pixels(img, (x+i, y+j), pixel_validator, pixel_set))
        return pixel_set


def get_blob_extremes(pixel_set):
    """
    Get the minimum and maximum x and y values for a given blob.
    """
    x_vals, y_vals = zip(*list(pixel_set))
    return np.min(x_vals), np.max(x_vals), np.min(y_vals), np.max(y_vals)


padding = 1

min_aspect_ratio = 1
max_aspect_ratio = 6

min_area = 50
max_area = 600

frame = (50, 20)

def blob_processor(pixel_set, base_img):#, tuft_processor):
    """
    Creates a blob dictionary from a pixel set.
    Returns `(True, blob)` if the blob is valid
    and `(False, None)` if not.
    """
    xmin, xmax, ymin, ymax = get_blob_extremes(pixel_set)

    dim_x = xmax - xmin# + 2*padding
    dim_y = ymax - ymin# + 2*padding

    if (dim_x == 0 or dim_y == 0):
        return False, None

    area = dim_x * dim_y
    aspect_ratio = dim_x / dim_y

    if (area < min_area or
        area > max_area or
        aspect_ratio < min_aspect_ratio or
            aspect_ratio > max_aspect_ratio):
        return False, None


    #original_img = np.ones(shape=(dim_x + 2*padding, dim_y + 2*padding)) * np.min(base_img[xmin:xmax, ymin:ymax])

    mean_pos_x, mean_pos_y = ((xmax + xmin)/2, (ymax + ymin)/2)
    mid_x = int(np.floor(mean_pos_x))
    mid_y = int(np.floor(mean_pos_y))
    pad_x_before = int(np.floor(frame[0]/2))
    pad_x_after = frame[0] - pad_x_before
    pad_y_before = int(np.floor(frame[1]/2))
    pad_y_after = frame[1] - pad_y_before


    original_img = np.zeros(shape=frame)
    original_img[:, :] = base_img[mid_x - pad_x_before : mid_x + pad_x_after,\
                                 mid_y - pad_y_before : mid_y + pad_y_after]
    #c_x, c_y = int(np.floor(dim_x / 2)), int(np.floor(dim_y / 2))
    #x_top, x_bot = f_cx - c_x, f_cx - c_x + dim_x
    #y_left, y_right = f_cy - c_y, f_cy - c_y + dim_y
    #original_img[:, :] = base_img[x_bot:x_top, y_left:y_right]
    #original_img[padding:-padding, padding:-padding] = base_img[xmin:xmax, ymin:ymax]

    #processed = tuft_processor(original_img)

    blob_entry = {
        #'img': processed,
        'img': original_img, #'tuft': original_img
        'mean_pos_x': mean_pos_x, #(xmax + xmin)/2,
        'mean_pos_y': mean_pos_y, #(ymax + ymin)/2,
        'dim_x': dim_x,
        'dim_y': dim_y,
        'area': area,
        'aspect_ratio': aspect_ratio,
        'corner_x': xmin,
        'corner_y': ymin,
    }

    return True, blob_entry


def clean_blob(img, pixel_set):
    """
    Sets all blob pixels in an image to 0.
    """
    img[index_set(pixel_set)] = 0

def index_set(pixel_set):
    """
    Create a Numpy-compatible index
    from a pixel set
    """
    return tuple(zip(*list(pixel_set)))


def create_dataframe(img, base_img, blob_processor):#, tuft_processor):
    """
    Scans a processed image and creates
    a dataframe of blob entries.
    """
    results = []
    process_img = img.copy()

    while (np.any(process_img == 1)):
        coords = np.unravel_index(
            np.argmax(process_img == 1), process_img.shape)
        pixel_set = get_blob_pixels(process_img, (coords[0], coords[1]), pixel_set=set())
        is_valid, blob = blob_processor(pixel_set, base_img)#, tuft_processor)
        if (is_valid):
            results.append(blob)
        clean_blob(process_img, pixel_set)

    results_df = pd.DataFrame(results)

    #results_df.drop(results_df.loc[results_df['img']])

    return results_df
