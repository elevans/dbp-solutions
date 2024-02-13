#@ CommandService cmd
#@ ConvertService cs
#@ OpService ops
#@ UIService ui
#@ ImgPlus image
#@ Double (label="Gaussian Sigma:", style="format:#.00", min=0.0, stepsize=0.1, value=6.00) sigma
#@ Double (label="Watershed Sigma:", style="format:#.00", min=0.0, stepsize=0.1, value=30.00) sigma
#@ Long (label="Mean radius:", style="format:#", min=0, stepsize=1, value=5) radius
#@output ImgPlus out

from ij.plugin.frame import RoiManager
from ij.gui import PolygonRoi
from net.imglib2.roi.labeling import LabelRegions, ImgLabeling
from net.imglib2.roi import Regions
from net.imglib2.algorithm.neighborhood import HyperSphereShape
from net.imglib2.algorithm.labeling.ConnectedComponents import StructuringElement

def gaussian_subtraction(img, sig):
    """Perform a gaussian subtraction.

    Blur the input image with the given sigma value in both
    horizontal and vertical axes. The returned image is the
    difference between the input and blurred image.

    :param img: Input ImgPlus image.
    :param sig: Sigma value for gaussian blur.
    :return: The difference between the input and blurred image.
    """

    img = ops.convert().float32(img)
    img_gauss = ops.filter().gauss(img, sig)
    img_sub = ops.math().subtract(img, img_gauss)

    return img_sub


def split_channels(img, ch_axis=2):
    """Split the input image along the given channel axis.

    Split an imput ImgPlus image along the given channel axis.

    :param img: Input ImgPlus image.
    :param ch_axis: Channel axis position.
    :return: List of split images.
    """
    channels = []
    for i in range(img.dimensionsAsLongArray()[ch_axis]):
        channels.append(ops.transform().hyperSliceView(img, ch_axis, i))
    
    return channels


def create_nuclei_labeling(img, thres="li"):
    """Create a nuclear binary mask.

    Description

    :param img:
    :param thres:
    :return:
    """
    # apply mean/multiply background supression
    img = ops.convert().float32(img)
    img_mean = ops.create().img(img)
    ops.filter().mean(img_mean, img, HyperSphereShape(radius))
    img_res = ops.math().multiply(img, img_mean)

    # create and clean up mask
    mask = ops.run("threshold.{}".format(thres), img_res)
    mask = ops.morphology().fillHoles(mask)

    # apply watershed
    labeling = ops.image().watershed(None, mask, True, False, 30.0, mask)

    return labeling


def create_puncta_labeling(img, thres="huang"):
    """Create a puncta binary mask.

    Description

    :param img:
    :param thres:
    :return:
    """
    # suppress background and threshold puncta
    img_sub = gaussian_subtraction(img, sigma)
    mask = ops.run("threshold.{}".format(thres), img_sub)
    
    # use morphology open operation
    shape = HyperSphereShape(2)
    mask = ops.morphology().open(mask, [shape])

    # create an ImgLabeling with CCA
    labeling = ops.labeling().cca(mask, StructuringElement.EIGHT_CONNECTED)

    return labeling


def create_nuc_puncta_labeling(puncta_labeling, nuc_labeling):
    # create label regions
    regs = LabelRegions(nuc_labeling)
    p_index_img = puncta_labeling.getIndexImg()
    vals = []
    for r in regs:
        sample = Regions.sample(r, p_index_img)
        # get the sample region's cursor
        c = sample.cursor()
        while c.hasNext():
            c.fwd()
            vals.append(int(c.get().toString()))
            # only keep unique values
            vals = list(set(vals))

    # only keep unique values
    vals = list(set(vals))

    # remove the background label
    if 0 in vals:
        vals.remove(0)

    # set non-nuclear puncta to 0
    c = p_index_img.cursor()
    while c.hasNext():
        c.fwd()
        if int(c.get().toString()) in vals:
            continue
        else:
            c.get().set(0)

    # return the filtered index image as a labeling
    return cs.convert(p_index_img, ImgLabeling)


def label_img_to_roi_manager(labeling):
    """Convert ImgLabeling to the ROIManager Rois.
    """
    contours = []
    regions = LabelRegions(labeling)
    # apply contour op to each region
    for r in regions:
        contours.append(ops.geom().contour(r, True))

    # add labels to the roi manager
    rm = RoiManager.getRoiManager()
    for c in contours:
        rm.addRoi(cs.convert(c, PolygonRoi))


# do the analysis
chs = split_channels(image)
p_labeling = create_puncta_labeling(chs[0])
n_labeling = create_nuclei_labeling(chs[1])
pn_labeling = create_nuc_puncta_labeling(p_labeling, n_labeling)
label_img_to_roi_manager(n_labeling)
label_img_to_roi_manager(pn_labeling)
