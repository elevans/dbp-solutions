#@ OpEnvironment ops
#@ OpService ijops
#@ ConvertService cs
#@ UIService ui
#@ Img (label = "Input image:", autofill = false) img
#@ String (visibility = MESSAGE, value ="<b>[ Channel selection settings ]</b>", required = false) ch_msg
#@ String (label = "Channel A name", value = "mCherry") ch_a_name
#@ String (label = "Channel B name", value = "YFP") ch_b_name
#@ String (visibility = MESSAGE, value ="<b>[ Filter settings ]</b>", required = false) fs_msg
#@ Double (label = "High pass filter Sigma:", style = "format:0.00", min = 0.0, value = 17.0) hp_sigma
#@ Integer (label = "Minimum puncta size (pixels):", min = 0, value = 0) min_size
#@ Integer (label = "Maximum puncta size (pixels):", min = 0, value = 0) max_size
#@ String (visibility = MESSAGE, value ="<b>[ Calibration settings ]</b>", required = false) cal_msg
#@ Float (label = "Image calibration", style = "format:0.0000", value = 0.13) cal
#@output Img output

from net.imglib2.algorithm.labeling.ConnectedComponents import StructuringElement
from net.imglib2.roi import Regions
from net.imglib2.roi.labeling import ImgLabeling, LabelRegions
from net.imglib2.type.logic import BitType
from net.imglib2.type.numeric.real import FloatType

from org.scijava.table import DefaultGenericTable

def split_img(image, axis=2):
    """Create views along a given axis.

    Create a view of the given image along a given axis.

    :param image:

        Input Img.

    :param axis:

        Axis to split along.

    :return:

        View of the input Img split along the given axis.
    """
    views = []
    for i in range(image.dimensionsAsLongArray()[axis]):
        views.append(ops.op("transform.hyperSliceView").input(image, axis, i).apply())

    return views


def gaussian_subtraction(image, sigma):
    """Perform a Gaussian subtraction on an image.

    Apply a Gaussian blur and subtract from input image.

    :param image:

        Input Img.

    :param sigma:

        Sigma value.

    :return:

        Gaussian blur subtracted image.
    """
    blur = ops.op("filter.gauss").input(image, sigma).apply()
    out = ops.op("create.img").input(image, FloatType()).apply()
    ops.op("math.sub").input(image, blur).output(out).compute()

    return out


def size_filter_labeling(labeling, min_size, max_size=None):
    """Apply a size filter to an ImgLabeling.

    Apply a size filter to an ImgLabeling. Labels that are within the minimum
    and maximum bounds provided are retained.

    :param labeling:

        Input ImgLabeling.

    :param min_size:

        Minimum number of label pixels. If None, then the minimum
        value is set to 0.

    :param max_size:

        Maximum number of label pixles. If None, then the maximum
        value is set to dim lengths X * Y.
    """
    # get label image from the labeling
    idx_img = labeling.getIndexImg()

    # set max/min defaults if necessary
    if not min_size:
        min_size = 0

    if not max_size:
        dims = idx_img.dimensionsAsLongArray()
        max_size = dims[0] * dims[1]

    # get a smple over the label image and remove outside size bounds
    regs = LabelRegions(labeling)
    for r in regs:
        s = Regions.sample(r, idx_img)
        l = int(ops.op("stats.size").input(s).apply())
        if l <= min_size or l >= max_size:
            remove_label(s)

    return cs.convert(idx_img, ImgLabeling)# process the data


def remove_label(sample):
    """Set a label to 0.

    Set the given sample region pixel values to 0.

    :param sample:

        A sample region.
    """
    # get the sample region's cursor
    c = sample.cursor()
    # set all pixels within the sample region to 0
    while c.hasNext():
        c.fwd()
        c.get().set(0)

# run the analysis
chs = split_img(img)
sub = gaussian_subtraction(chs[1], hp_sigma)
ths = ops.op("create.img").input(sub, BitType()).apply()
ops.op("threshold.triangle").input(sub).output(ths).compute()
labeling = ops.op("labeling.cca").input(ths, StructuringElement.EIGHT_CONNECTED).apply()

# apply size filter to the labeling
labeling = size_filter_labeling(labeling, min_size, max_size)

# compute stats and populate table
regs = LabelRegions(labeling)
regs_labels = regs.getExistingLabels()

table = DefaultGenericTable(4, 0)
table.setColumnHeader(0, "label")
table.setColumnHeader(1, "size (um^2)")
table.setColumnHeader(2, "{} mfi".format(ch_a_name))
table.setColumnHeader(3, "{} mfi".format(ch_b_name))

i = 0
for r in regs:
    s1 = Regions.sample(r, chs[0])
    s2 = Regions.sample(r, chs[1])
    table.appendRow()
    table.set("label", i, int(r.getLabel()))
    table.set("size (um^2)", i, ops.op("stats.size").input(s1).apply() * cal)
    table.set("{} mfi".format(ch_a_name), i, ijops.stats().mean(s1).getRealDouble())
    table.set("{} mfi".format(ch_b_name), i, ijops.stats().mean(s2).getRealDouble())
    i += 1

ui.show("result table", table)
output = labeling.getIndexImg()
