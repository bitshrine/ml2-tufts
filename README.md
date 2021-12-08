# ML4Science - Flow detachment analysis

> Project done as part of the CS-433 Machine Learning course at EPFL

**Team:**
- Kepler Warrington-Arroyo ([@bitshrine](https://github.com/bitshrine))
- César Descalzo (add your GitHub name here)
- Zied Mustapha ([@Laniakea1999](https://github.com/Laniakea1999))

___

## 0. Preliminary processing
Image processing is done by [`img_processing.py`](scripts/img_processing.py).
> **Notes:**
> - This requires `numpy`, `scipy`, `pillow`, and `pandas`.
> - The script should be run from the `scripts` folder, and a folder named `output` should exist in the root directory, to prevent an error being thrown when writing the output.

**Usage:**
```
python3 img_processing.py -s img_source -w -a
```

**Options:**
- `-s` or `--source`: used to specify the path of the source `.tiff` image.
- `-w` or `--whole`: if this is specified, the script generates an image which displays the detected tufts on the original image. This is useful to know which tufts were detected and how they were processed.
- `-a` or `--all`: if this is specified, generates `.tiff` images for each of the detected tufts.

**Output:**

The script produces a .json file which can be loaded into a Pandas DataFram (using `pd.read_json()` for instance). The features in the DataFrame are:
| img | tuft | mean_pos_x | mean_pos_y | dim_x | dim_y | area | aspect_ratio | corner_x | corner_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image of the tuft | Processed image of the tuft | Center along the x axis | Center along the y axis | Width of the image | Length of the image | Area of the image in pixels | Aspect ratio of the image | x-coordinate of the upper left-hand corner of the image | y-coordinate of the upper right-hand corner of the image |