import numpy as np
from scipy import ndimage
from PIL import Image

import sys, getopt

from basic import *
from blobs import *


def recomp_img(df, img):
    """
    Print all the tuft sub-images onto the original image.
    Use this to visualize which tufts were detected by the pipeline
    and how the pipeline has processed them.
    """
    img = img.copy()

    for index, tuft in df.iterrows():
        img[tuft['corner_x']:tuft['corner_x']+tuft['dim_x'],
            tuft['corner_y']:tuft['corner_y']+tuft['dim_y']] = tuft['tuft']

    return img


def pipeline(src, whole, all_tufts):
    print("Processing {img}".format(img=src))

    img = np.array(Image.open(src)).astype(np.float32)

    # Crop
    img = img[245:575, 30:-20]

    # Rescale
    img = rescale_image(img)

    wing_base = img.copy()

    # Apply tophat transformation
    img = 1 - img
    th_size = (10, 10)
    img = ndimage.white_tophat(img, size=th_size)

    # Better separate values
    img = separate(img, 6)

    # Treshold image
    img = threshold(img, 0.1)

    # Extract blobs
    img = create_blobs(img, 2, 0.35)

    # Create dataframe
    blobs_df = create_dataframe(img, wing_base, blob_processor)

    out_name = src.split('.')[-2].split('/')[-1]
    blobs_df.to_json('../output/{name}.json'.format(name=out_name))

    if (whole):
        whole_img = recomp_img(blobs_df, wing_base)
        whole_img = Image.fromarray(whole_img)
        whole_img.save('../output/{name}_whole.tiff'.format(name=out_name))

    if (all_tufts):
        for ind, entry in blobs_df.iterrows():
            tuft_img = Image.fromarray(entry['img'])
            tuft_img.save('../output/{name}_{i}.tiff'.format(name=out_name, i=ind))




opts = 's:aw'
longopts = ['source=', 'whole', 'all']
if (__name__ == "__main__"):
    try:
        opts, args = getopt.getopt(sys.argv[1:], opts, longopts)#, ["help", "output="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)

    src = None
    all_tufts = False
    whole = False

    for o, a in opts:
        if o in ("-s", "--source"):#, "--help"):
            src = a
        elif o in ("-w", "--whole"):
            whole = True
        elif o in ("-a", "--all"):
            all_tufts = True
        else:
            assert False, "unhandled option"

    pipeline(src, whole, all_tufts)
