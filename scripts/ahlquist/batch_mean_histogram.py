#@ OpEnvironment ops
#@ OpService ijops
#@ IOService io
#@ DatasetIOService ds
#@ ConvertService cs
#@ UIService ui
#@ File (label = "Input directory", style="directory") in_dir
#@ String (label = "File extension", value=".tif") ext
#@ String (choices={"otsu", "huang", "moments", "renyiEntropy", "triangle", "yen"}, style="listBox") thres_method
#@ String (visibility=MESSAGE, value="<b>Channel configuration</b>", required=false) ch_msg
#@ String (label = "Reference channel name", value="p16") ch_a_name
#@ Integer (label = "Reference channel number", min = 1, value=1) ch_a
#@ String (label = "Comparison channel name", value="MUC4") ch_b_name
#@ Integer (label = "Comparison channel number", min = 1, value=2) ch_b
#@ Boolean (label = "Create stack with masks", value=False) save

import os

from net.imglib2 import FinalDimensions
from net.imglib2.img.array import ArrayImgs
from net.imglib2.type.numeric.integer import UnsignedShortType
from net.imglib2.type.logic import BitType
from net.imglib2.roi import Regions
from net.imglib2.roi.labeling import ImgLabeling, LabelRegions
from net.imglib2.view import StackView, Views

from org.scijava.table import DefaultGenericTable

from java.lang import Short
from java.lang import Long
from jarray import array

def create_mask_stack(images, labelings_arr):
    """Create a new stack with the mask as a channel.

    Creates a list of stacks where the input images are
    concatenated with their respective masks (set to
    65,535 pixel value).

    :param images:

        A list of Imgs.

    :param labelings_arr:

        A list of ImgLabelings.

    :return:

        A list of Imgs with masks added as a channel.
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
        stacks.append(Views.concatenate(2, StackView(st)))

    return stacks


def set_max_pixel_value(sample):
    """Set the maximum pixel value for uint16.

    Set all pixels within the given sample to the max
    value of the 16-bit unsigned data range (65,535).

    :param sample:

        An ImgLib2 SamplingInterableInterval
    """
    # get the cursor
    c = sample.cursor()
    while c.hasNext():
        c.fwd()
        c.get().set(65535)


def count_pixels(sample):
    """Count the number of > 0 pixels in an ImgLib2 sample.

    Count the number of pixels that have a value greater than
    0.

    :param sample:

        An ImgLib2 SamplingIterableInterval.

    :return:

        The number of greater than 0 pixels as an integer.
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

    return p


def extract_channel(image, ch):
    """Extract a given channel as a view.

    Extract the given channel from the input image.
    Note that this function creates a copy of the data.

    :param image:

        Input Img.

    :param ch:

        Channel number to extract.

    :return: An ArrayImg with the channel data.
    """
    return ops.op("transform.hyperSliceView").input(image, 2, ch - 1).apply()


def extract_file_names(paths):
    """Extract file names from a list of paths.

    Extract the file names from a list of file paths.

    :param paths:

        A list of file paths.

    :return:

        A list of file names.
    """
    names = []
    for p in paths:
        n = os.path.basename(p)
        names.append(os.path.splitext(n)[0])

    return names


def get_file_paths(base_dir, ext):
    """Get all file paths in a given directory.

    Get all file paths in the provided directory with
    the given extension.

    :param base_dir:

        The base directory to search in.

    :param ext:

        The extension for considered files (e.g. ".tif")

    :return:

        A list of file paths.
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
    """Load images from a list of file paths.

    Load images from a list of file paths as Imgs.

    :param paths:

        A list of file paths.

    :return:

        A list of Imgs.
    """
    images = []
    for p in paths:
        print("[INFO]: Loading image {}".format(os.path.basename(p)))
        images.append(ds.open(p))

    return images


def mean(vals):
    """Compute the mean of a list of values.

    Compute the mean from a list/sequence of values (e.g. integers).

    :param vals:

        A list of values (e.g. integers/floats)

    :return:

        The mean as a float.
    """
    return sum(vals) / float(len(vals))


def mean_histogram(images, bins=65536):
    """Compute the mean histogram of a list of images.

    Compute the mean histogram from a list of images by extracting
    the histograms from each image in the list and averaging them.
    The produced mean histogram is then converted into an ImgLib2
    Histogram1d type.

    :param images:

        A list of Imgs.

    :param bins:

        The number of histogram bins. Typically the bit depth of the
        input image (e.g. 16-bit/65,536).

    :return:

        An ImgLib2 Histogram1d.
    """
    hists = []
    # get individual image histograms
    for img in images:
        hists.append(ops.op("image.histogram").input(img, bins).apply().toLongArray())

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

    Process the batch of images and populate a results table.

    :param image_paths:

        A path of image files.

    :return:

        A SciJava results table.
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
    print("[INFO]: Computing mean histogram...")
    mean_hist_a = mean_histogram(imgs_a)
    mean_hist_b = mean_histogram(imgs_b)

    # compute thresholds and apply
    print("[INFO]: Computing thresholds...")
    thres_a = ops.op("threshold.otsu").input(mean_hist_a).apply()
    thres_b = ops.op("threshold.otsu").input(mean_hist_b).apply()
    masks_a = []
    masks_b = []
    masks_xor = []
    print("[INFO]: Applying thresholds...")
    for i in range(count):
        m_a = ops.op("create.img").input(imgs_a[i], BitType()).apply()
        m_b = ops.op("create.img").input(imgs_b[i], BitType()).apply()
        m_x = ops.op("create.img").input(imgs_a[i], BitType()).apply()
        ops.op("threshold.apply").input(imgs_a[i], thres_a).output(m_a).compute()
        ops.op("threshold.apply").input(imgs_b[i], thres_b).output(m_b).compute()
        ops.op("logic.xor").input(m_a, m_b).output(m_x).compute()
        masks_a.append(m_a)
        masks_b.append(m_b)
        masks_xor.append(m_x)

    # create ImgLabelings
    print("[INFO]: Creating labels...")
    labelings_a = []
    labelings_b = []
    labelings_xor = []
    for i in range(count):
        labelings_a.append(cs.convert(masks_a[i], ImgLabeling))
        labelings_b.append(cs.convert(masks_b[i], ImgLabeling))
        labelings_xor.append(cs.convert(masks_xor[i], ImgLabeling))
    
    # create table with headers
    table = DefaultGenericTable(9, 0)
    table.setColumnHeader(0, "name")
    table.setColumnHeader(1, "{}_roi_size".format(ch_a_name))
    table.setColumnHeader(2, "{}_roi_mfi".format(ch_a_name))
    table.setColumnHeader(3, "{}_threshold_value".format(ch_a_name))
    table.setColumnHeader(4, "{}_size_in_p16_roi".format(ch_b_name))
    table.setColumnHeader(5, "{}_mfi_in_p16_roi".format(ch_b_name))
    table.setColumnHeader(6, "{}_mfi_outside_p16_roi".format(ch_b_name))
    table.setColumnHeader(7, "{}_roi_size".format(ch_b_name))
    table.setColumnHeader(8, "{}_threshold_value".format(ch_b_name))

    for i in range(count):
        # extract label regions
        regs_a = LabelRegions(labelings_a[i])
        regs_b = LabelRegions(labelings_b[i])
        regs_xor = LabelRegions(labelings_xor[i])

        # extract region -- only one label should be in the labeling
        r_a = regs_a.getLabelRegion(1)
        r_b = regs_b.getLabelRegion(1)
        r_x = regs_xor.getLabelRegion(1)

        # create samples
        sample_a = Regions.sample(r_a, imgs_a[i]) # p16 label w/ p16 channel
        sample_b = Regions.sample(r_b, imgs_b[i]) # MUC4 label w/ MUC4 channel
        sample_ab = Regions.sample(r_a, imgs_b[i]) # p16 label w/ MUC4 channel
        sample_ab_m = Regions.sample(r_a, labelings_b[i].getIndexImg()) # p16 label w/ MUC4 mask
        sample_xor = Regions.sample(r_x, imgs_b[i]) # XOR mask w/ MUC4 channel

        # compute stats and write to table
        table.appendRow()
        table.set("name", i, names[i])
        table.set("{}_roi_size".format(ch_a_name), i, ops.op("stats.size").input(sample_a).apply())
        table.set("{}_roi_mfi".format(ch_a_name), i, ijops.stats().mean(sample_a).getRealDouble())
        table.set("{}_threshold_value".format(ch_a_name), i, thres_a)
        table.set("{}_size_in_p16_roi".format(ch_b_name), i, count_pixels(sample_ab_m))
        table.set("{}_mfi_in_p16_roi".format(ch_b_name), i, ijops.stats().mean(sample_ab).getRealDouble())
        table.set("{}_mfi_outside_p16_roi".format(ch_b_name), i, ijops.stats().mean(sample_xor).getRealDouble())
        table.set("{}_roi_size".format(ch_b_name), i, ops.op("stats.size").input(sample_b).apply())
        table.set("{}_threshold_value".format(ch_b_name), i, thres_b)

    if save:
        print("[INFO]: Creating mask stacks...")
        stacks = create_mask_stack(imgs, (labelings_a, labelings_b, labelings_xor))
        for i in range(len(stacks)):
            print("[INFO]: Saving image {}_mask_stack.tif...".format(names[i]))
            io.save(stacks[i], os.path.join(in_dir.toString(), "{}_mask_stack.tif".format(names[i])))

    return table

# run image analysis
img_paths = get_file_paths(in_dir, ext)
result_table = process_batch(img_paths)
ui.show(result_table)
print("[INFO]: Done!")
