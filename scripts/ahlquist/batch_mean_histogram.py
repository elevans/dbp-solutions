#@ OpEnvironment ops
#@ OpService ijops
#@ IOService io
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
from net.imglib2.view import Views

from org.scijava.table import DefaultGenericTable

from java.lang import Short
from java.lang import Long
from jarray import array

def create_mask_stack(images, labelings_arr):
    """Save masks as a stack by appending to the channel axis.
    """
    count = len(images)
    stacks = []
    for i in range(count):
        st = [images[i]] # the base/input image
        for ch in labelings_arr:
            # duplicate index image and conver to 16-bit
            msk = ops.op("copy.img").input(ch[i].getIndexImg()).apply()
            msk = ops.op("convert.uint16").input(msk).apply()
            # create a sample and only set those pixels to max
            regs = LabelRegions(ch[i])
            r = regs.getLabelRegion(1) # there is only one region
            smp = Regions.sample(r, msk)
            # set the pixels within the sample to 65535
            set_max_pixel_value(smp)
            st.append(msk)

        # concatenate view along the same axis
        stacks.append(Views.concatenate(2, st))

    return stacks


def set_max_pixel_value(sample):
    """Set the maximum pixel value to 65535 for uint16.
    """
    # get the cursor
    c = sample.cursor()
    while c.hasNext():
        c.fwd()
        c.get().set(65535)


def count_pixels(sample):
    """Count the number of > 0 pixels in an ImgLib2 sample.
    """
    # get the cursor
    c = sample.cursor()
    p = 0
    while c.hasNext():
        c.fwd()
        if c.get().getRealDouble() > 0:
            p += 1
        else:
            continue

    return float(p)


def extract_channel(image, ch):
    """Extract a given channel as a view.
    """
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
    print("[INFO]: Extracting channels...")
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
    print("[INFO]: Creating labels...")
    labelings_a = []
    labelings_b = []
    for i in range(count):
        labelings_a.append(cs.convert(masks_a[i], ImgLabeling))
        labelings_b.append(cs.convert(masks_b[i], ImgLabeling))
    
    # create table with headers
    table = DefaultGenericTable(7, 0)
    table.setColumnHeader(0, "name")
    table.setColumnHeader(1, "p16_roi_size")
    table.setColumnHeader(2, "p16_roi_mfi")
    table.setColumnHeader(3, "p16_threshold_value")
    table.setColumnHeader(4, "MUC4_size_in_p16_roi")
    table.setColumnHeader(5, "MUC4_mfi_in_p16_roi")
    table.setColumnHeader(6, "MUC4_threshold_value")

    for i in range(count):
        # extract label regions
        regs_a = LabelRegions(labelings_a[i])

        # extract region -- only one label should be in the labeling
        r_a = regs_a.getLabelRegion(1)

        # create samples
        sample_a = Regions.sample(r_a, imgs_a[i]) # p16 label w/ p16 channel
        sample_ab = Regions.sample(r_a, imgs_b[i]) # p16 label w/ MUC4 channel
        sample_ab_m = Regions.sample(r_a, labelings_b[i].getIndexImg()) # p16 label w/ MUC4 mask

        # compute stats and write to table
        table.appendRow()
        table.set("name", i, names[i])
        table.set("p16_roi_size", i, ijops.stats().size(sample_a).getRealDouble())
        table.set("p16_roi_mfi", i, ijops.stats().mean(sample_a).getRealDouble())
        table.set("p16_threshold_value", i, thres_a)
        table.set("MUC4_size_in_p16_roi", i, count_pixels(sample_ab_m))
        table.set("MUC4_mfi_in_p16_roi", i, ijops.stats().mean(sample_ab).getRealDouble())
        table.set("MUC4_threshold_value", i, thres_b)

    if save:
        print("[INFO]: Creating mask stacks...")
        stacks = create_mask_stack(imgs, (labelings_a, labelings_b))
        for i in range(len(stacks)):
            print("[INFO]: Saving image {}_mask_stack.tif...".format(names[i]))
            io.save(stacks[i], os.path.join(in_dir.toString(), "{}_mask_stack.tif".format(names[i])))

    return table

# run image analysis
img_paths = get_file_paths(in_dir, ext)
result_table = process_batch(img_paths)
ui.show(result_table)
