#@ OpEnvironment ops
#@ OpService ijops
#@ UIService ui
#@ ImgPlus img
#@ String (label = "Channel A name", value = "mCherry") ch_a_name
#@ String (label = "Channel B name", value = "YFP") ch_b_name
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

# process the data
chs = split_img(img)
sub = gaussian_subtraction(chs[1], 17.0)
ths = ops.op("create.img").input(sub, BitType()).apply()
ops.op("threshold.triangle").input(sub).output(ths).compute()
labeling = ops.op("labeling.cca").input(ths, StructuringElement.EIGHT_CONNECTED).apply()

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
