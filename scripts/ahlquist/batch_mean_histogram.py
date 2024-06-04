#@ OpEnvironment ops
#@ OpService ijops
#@ DatasetIOService ds
#@ ConvertService cs
#@ UIService ui
#@ File (label = "Input directory", style="directory") in_dir
#@ String (label = "File extension", value=".tif") ext
#@ String (choices={"otsu", "huang", "moments", "renyiEntropy", "triangle", "yen"}, style="listBox") thres_method
#@ Integer (label = "p16 Channel", value=1) ch_a
#@ Integer (label = "MUC4 Channel", value=2) ch_b
#@ Boolean (label = "Create stack with masks", value=False) save

import os

from net.imglib2.img.array import ArrayImgs
from net.imglib2.type.logic import BitType
from net.imglib2.roi import Regions
from net.imglib2.roi.labeling import ImgLabeling, LabelRegions

from org.scijava.table import DefaultGenericTable

from java.lang import Short
from jarray import array

def create_mask_stack(images, masks):
    # get list of input images
    # get list of masks (channel a, b)
    # convert masks to 16-bit (i.e. input type)
    # optional: multiply by 257 to max out mask value
    # stack view
    # return stack view
    return


def extract_channel(image, ch):
    """Extract a given channel as a view.
    """
    print("[INFO]: Extracting channel {}".format(ch))
    dims = image.dimensionsAsLongArray()
    min_interval = array([0, 0, ch -1], "l")
    max_interval = array([dims[0] - 1, dims[1] - 1, ch - 1], "l")
    return ops.op("transform.intervalView").input(image, min_interval, max_interval).apply()


def extract_file_names(paths):
    """Extract file names from a list of paths.
    """
    names = []
    for p in paths:
        n = os.path.basename(p)
        names.append(os.path.splitext(n)[0])

    return names


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


def load_images(paths):
    """Load images.
    """
    images = []
    for p in paths:
        print("[INFO]: Loading image {}".format(os.path.basename(p)))
        images.append(ds.open(p))

    return images


def mean(vals):
    """Compute the mean of a list of numbers.
    """
    return sum(vals) / float(len(vals))


def mean_histogram(images, bins=65536):
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
    
    # convert to imglib2 uint16 Histogram1d 
    mean_hist_arr = ArrayImgs.doubles(mean_hist, bins) # creates an Iterable<RealType>
    mean_hist_arr = ops.op("convert.uint16").input(mean_hist_arr).apply()

    return ops.op("image.histogram").input(mean_hist_arr, bins).apply()


def process_batch(image_paths):
    """Process the batch of images.
    """
    # load images
    imgs = load_images(image_paths)
    count = len(imgs)
    names = extract_file_names(image_paths)

    # extract channels
    imgs_a = []
    imgs_b = []
    for im in imgs:
        imgs_a.append(extract_channel(im, ch_a))
        imgs_b.append(extract_channel(im, ch_b))

    # create mean histograms
    mean_hist_a = mean_histogram(imgs_a)
    mean_hist_b = mean_histogram(imgs_b)

    # compute thresholds and apply
    print("[INFO]: Computing thresholds...")
    thres_a = ops.op("threshold.otsu").input(mean_hist_a).apply()
    thres_b = ops.op("threshold.otsu").input(mean_hist_b).apply()
    masks_a = []
    masks_b = []
    print("[INFO]: Applying thresholds...")
    for i in range(count):
        m_a = ops.op("create.img").input(imgs_a[i], BitType()).apply()
        m_b = ops.op("create.img").input(imgs_b[i], BitType()).apply()
        ops.op("threshold.apply").input(imgs_a[i], thres_a).output(m_a).compute()
        ops.op("threshold.apply").input(imgs_b[i], thres_b).output(m_b).compute()
        masks_a.append(m_a)
        masks_b.append(m_b)

    # create ImgLabelings
    labelings_a = []
    labelings_b = []
    for i in range(count):
        labelings_a.append(cs.convert(masks_a[i], ImgLabeling))
        labelings_b.append(cs.convert(masks_b[i], ImgLabeling))
    
    # create table with headers
    table = DefaultGenericTable(7, 0)
    table.setColumnHeader(0, "name")
    table.setColumnHeader(1, "p16_area")
    table.setColumnHeader(2, "p16_mfi")
    table.setColumnHeader(3, "p16_thres_value")
    table.setColumnHeader(4, "MUC4_area")
    table.setColumnHeader(5, "MUC4_mfi")
    table.setColumnHeader(6, "MUC4_thres_value")

    for i in range(count):
        # extract label regions
        regs_a = LabelRegions(labelings_a[i])
        regs_b = LabelRegions(labelings_b[i])

        # extract region -- only one label should be in the labeling
        r_a = regs_a.getLabelRegion(1)
        r_b = regs_b.getLabelRegion(1)

        # create samples
        sample_a = Regions.sample(r_a, imgs_a[i])
        sample_b = Regions.sample(r_b, imgs_b[i])

        # compute stats and write to table
        table.appendRow()
        table.set("name", i, names[i])
        table.set("p16_area", i, ijops.stats().size(sample_a).getRealDouble())
        table.set("p16_mfi", i, ijops.stats().mean(sample_a).getRealDouble())
        table.set("p16_thres_value", i, thres_a)
        table.set("MUC4_area", i, ijops.stats().size(sample_b).getRealDouble())
        table.set("MUC4_mfi", i, ijops.stats().mean(sample_b).getRealDouble())
        table.set("MUC4_thres_value", i, thres_b)

    # TODO: save hyper stacks w/ masks
    return table

# load batch images
img_paths = get_file_paths(in_dir, ext)

# debug stuff
x = process_batch(img_paths)
