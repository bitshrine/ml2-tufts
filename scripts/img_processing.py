import os

import numpy as np
from scipy import ndimage
from PIL import Image

import sys, getopt

from basic import *
from blobs import blob_processor, create_blobs, create_dataframe, frame
from tuft_helpers import tuft_processor


def recomp_img(df, img):
    """
    Print all the tuft sub-images onto the original image.
    Use this to visualize which tufts were detected by the pipeline
    and how the pipeline has processed them.
    """
    img = img.copy()
    img = np.squeeze(np.stack((img,) * 3, -1))

    for index, tuft in df.iterrows():
        tuft_img = np.array(tuft['img'])

        corner_x = tuft['corner_x'] - int(np.floor((tuft_img.shape[0] - tuft['dim_x']) / 2))
        corner_y = tuft['corner_y'] - int(np.floor((tuft_img.shape[1] - tuft['dim_y']) / 2))

        # Draw rescaled image
        img[corner_x:corner_x + tuft_img.shape[0], 
            corner_y:corner_y + tuft_img.shape[1], :] = np.squeeze(np.stack((tuft['img'],) * 3, -1))  # [1:-1, 1:-1]

        # Draw rectangle
        img[corner_x:corner_x + tuft_img.shape[0], corner_y, :] = np.array([0, 1, 0])
        img[corner_x:corner_x + tuft_img.shape[0],
            corner_y + tuft_img.shape[1], :] = np.array([0, 1, 0])

        img[corner_x,
            corner_y:corner_y + tuft_img.shape[1]:] = np.array([0, 1, 0])
        
        img[corner_x + tuft_img.shape[0],
            corner_y:corner_y + tuft_img.shape[1], :] = np.array([0, 1, 0])

    return Image.fromarray(np.uint8(img * 255)).convert('RGB')



def pad_img(img, frame):
    pad_x = img.shape[0] + frame[0] * 2
    pad_y = img.shape[1] + frame[1] * 2
    padding = np.zeros(shape=(pad_x, pad_y))
    padding[frame[0]:-frame[0], frame[1]:-frame[1]] = img

    return padding




def pipeline(src, whole, all_tufts):
    if (not os.path.exists(src)):
        print("Source image file does not exist!")
    print("Processing {img}".format(img=src))

    img = np.array(Image.open(src)).astype(np.float32)

    # Crop
    img = img[245:575, 30:-20]

    # Rescale
    img = rescale_image(img)

    wing_base = pad_img(img, frame)

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

    # Add padding for dataframe creation
    img = pad_img(img, frame)

    # Create dataframe
    blobs_df = create_dataframe(img, wing_base, blob_processor)#, tuft_processor)

    out_name = src.split('.')[-2].split('/')[-1]
    blobs_df.to_json('../output/{name}.json'.format(name=out_name))

    if (whole):
        whole_img = recomp_img(blobs_df, wing_base)
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

    #pipeline('../data/im0007.tiff', True, True)
    pipeline(src, whole, all_tufts)
