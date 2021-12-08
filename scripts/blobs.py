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


def coordinates_are_valid(img, coords):
    """
    Checks whether the given coordinates are inside the
    image boundaries
    """
    x, y = coords
    return (x >= 0 and x < len(img) and y >= 0 and y < len(img[0]))


coord_comb = [(-1, -1), (-1,  0), (-1,  1),
              (0, -1),           (0,  1),
              (1, -1), (1,  0), (1,  1)]


def get_blob_pixels(img, coords, pixel_set=set()):
    """Returns a set of pixel coordinates which belong to a blob"""
    x, y = coords
    if (img[x, y] == 0) or ((x, y) in pixel_set):
        return set()
    else:
        pixel_set.add((x, y))
        for i, j in coord_comb:
            if (coordinates_are_valid(img, (x+i, y+j))):
                pixel_set.update(get_blob_pixels(img, (x+i, y+j), pixel_set))
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


def blob_processor(pixel_set, img, base_img):
    """
    Creates a blob dictionary from a pixel set.
    Returns `(True, blob)` if the blob is valid
    and `(False, None)` if not.
    """
    xmin, xmax, ymin, ymax = get_blob_extremes(pixel_set)

    dim_x = xmax - xmin + 2*padding
    dim_y = ymax - ymin + 2*padding

    area = dim_x * dim_y
    aspect_ratio = dim_x / dim_y

    if (area < min_area or
        area > max_area or
        aspect_ratio < min_aspect_ratio or
            aspect_ratio > max_aspect_ratio):
        return False, None

    bb = (xmin - padding, xmax + padding, ymin - padding, ymax + padding)

    blob_entry = {
        'img': base_img[bb[0]:bb[1], bb[2]:bb[3]],
        'tuft': img[bb[0]:bb[1], bb[2]:bb[3]],
        'mean_pos_x': (xmax + xmin)/2,
        'mean_pos_y': (ymax + ymin)/2,
        'dim_x': dim_x,
        'dim_y': dim_y,
        'area': area,
        'aspect_ratio': aspect_ratio,
        'corner_x': xmin - padding,
        'corner_y': ymin - padding,
    }

    return True, blob_entry


def clean_blob(img, pixel_set):
    """
    Sets all blob pixels in an image to 0.
    """
    img[tuple(zip(*list(pixel_set)))] = 0


def create_dataframe(img, base_img, blob_processor):
    """
    Scans a processed image and creates
    a dataframe of blob entries.
    """
    results = []
    process_img = img.copy()

    while (np.any(process_img == 1)):
        coords = np.unravel_index(
            np.argmax(process_img == 1), process_img.shape)
        pixel_set = get_blob_pixels(process_img, (coords[0], coords[1]), set())
        is_valid, blob = blob_processor(pixel_set, img, base_img)
        if (is_valid):
            results.append(blob)
        clean_blob(process_img, pixel_set)

    results_df = pd.DataFrame(results)

    return results_df
