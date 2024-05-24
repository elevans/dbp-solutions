#@ OpEnvironment ops
#@ DatasetIOService ds
#@ ConvertService cs
#@ UIService ui
#@ File (label = "Input directory", style="directory") in_dir
#@ String (label = "File extension", value=".tif") ext
#@ String (choices={"otsu", "huang", "moments", "renyiEntropy", "triangle", "yen"}, style="listBox") thres_method
#@ Integer (label = "Channel", value=1) ch

import os
from net.imglib2.img.array import ArrayImgs
from jarray import array

def get_file_paths(base_dir, ext):
    """Get all file paths as in a given directory.
    """
    paths = []
    abs_dir = base_dir.getAbsolutePath()
    for root, dirs, file_names in os.walk(abs_dir):
        file_names.sort()
        for name in file_names:
            # check for file extension
            if not name.endswith(ext):
                continue
            paths.append(os.path.join(root, name))

    return paths


def load_images(paths, ch):
    """Load images.
    """
    images = []
    for p in paths:
        print("[IFNO]: Loading image {}".format(os.path.basename(p)))
        data = ds.open(p)
        dims = data.dimensionsAsLongArray()
        min_interval = array([0, 0, ch -1], "l")
        max_interval = array([dims[0] - 1, dims[1] - 1, ch - 1], "l")
        # create a view for each channel
        view = ops.op("transform.intervalView").input(data, min_interval, max_interval).apply()
        images.append(view)

    return images


def mean(vals):
    """Compute the mean of an array of numbers.
    """
    return sum(vals) / float(len(vals))


def mean_histograms(images, bins=65536):
    """Compute the mean histogram of a list of images.
    """
    print("[INFO]: Calculating individual histograms...")
    hists = []
    # get individual image histograms
    for img in images:
        hists.append(ops.op("image.histogram").input(img, bins).apply().toLongArray())

    print("[INFO]: Calculating mean histogram...")
    mean_hist = []
    temp = []
    for b in range(bins):
        del temp[:] # clear the list -- avoid creating new ones
        for h in hists:
            temp.append(h[b])
        mean_hist.append((mean(temp)))
    
    # convert to imglib2 Histogram1d
    mean_hist_arr = ArrayImgs.doubles(mean_hist, bins) # creates an Iterable<RealType>
     
    return ops.op("image.histogram").input(mean_hist_arr, bins).apply()

img_paths = get_file_paths(in_dir, ext)
imgs = load_images(img_paths, ch)
result = mean_histograms(imgs)

# debug stuff
print(type(result))
