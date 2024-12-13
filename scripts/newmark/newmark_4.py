#@ OpEnvironment ops
#@ ConvertService cs
#@ IOService io
#@ UIService ui
#@ Img (label = "Input image:", autofill = false) img
#@ String (visibility = MESSAGE, value ="<b>[ Channel selection settings ]</b>", required = false) ch_msg
#@ String (label = "Puncta channel number:", choices = {"None", "1", "2", "3", "4", "5"}, style = "listBox", value = "None") p_ch
#@ String (label = "Nuclear channel number:", choices = {"None", "1", "2", "3", "4", "5"}, style = "listBox", value = "None") n_ch
#@ String (visibility = MESSAGE, value ="<b>[ Puncta segmentation settings ]</b>", required = false) pun_msg
#@ Double (label = "High pass filter Sigma:", style = "format:0.00", min = 0.0, value = 5.0) hp_sigma
#@ Double (label = "Watershed Sigma:", style = "format:0.00", min = 0.0, value = 2.0) ws_sigma
#@ String (label = "Global threshold method:", choices={"huang", "ij1", "intermodes", "isoData", "li", "maxEntropy", "maxLikelihood", "mean", "minError", "minimum", "moments", "otsu", "percentile", "renyiEntropy", "rosin", "shanbhag", "triangle", "yen"}, style="listBox") threshold
#@ Integer (label = "Minimum puncta size (pixels):", min = 0, value = 0) min_size
#@ Integer (label = "Maximum puncta size (pixels):", min = 0, value = 0) max_size
#@ Boolean (label = "Show puncta label image:", value = false) pun_show
#@ String (visibility = MESSAGE, value ="<b>[ Cellpose nuclear segmentation settings ]</b>", required = false) cp_msg
#@ String (label = "Cellpose Python path:") cp_py_path 
#@ String (label = "Pretrained model:", choices = {"cyto", "cyto2", "cyto2_cp3", "cyto3", "nuclei", "livecell_cp3", "deepbacs_cp3", "tissuenet_cp3", "bact_fluor_cp3", "bact_phase_cp3", "neurips_cellpose_default", "neurips_cellpose_transformer", "neurips_grayscale_cyto2", "transformer_cp3", "yeast_BF_cp3", "yeast_PhC_cp3"}, style="listBox") cp_model
#@ Float (label = "Diameter:", style = "format:0.00", value=0.00, stepSize=0.01) cp_diameter
#@ Boolean (label = "Enable 3D segmentation:", value = True) use_3d
#@ Boolean (label = "Show nuclear label image:", value = false) nuc_show
#@output Img output

import os
import subprocess
import shutil

from net.imglib2.algorithm.labeling.ConnectedComponents import StructuringElement
from net.imglib2.algorithm.neighborhood import HyperSphereShape
from net.imglib2.roi import Regions
from net.imglib2.roi.labeling import ImgLabeling, LabelRegions
from net.imglib2.type.logic import BitType
from net.imglib2.type.numeric.real import FloatType

from ij.gui import PointRoi
from ij.plugin.frame import RoiManager

from java.nio.file import Files

def cca(image, se):
    """Run connecated component analysis.

    Run connecated component analysis (CCA) on the given binary image with
    the selected structuring element.

    :param image:

        Input image.

    :param se:

        String ("FOUR" or "EIGHT") indicating the type of structuring element.

    :param:

        An ImgLabeling of the input image.
    """
    # get the structuring element
    if se == "FOUR":
        strel = StructuringElement.FOUR_CONNECTED
    else:
        strel = StructuringElement.EIGHT_CONNECTED

    return ops.op("labeling.cca").input(image, strel).apply()


def cellpose(image):
    """Run cellpose with the given settings.

    Run Cellpose with the given settings and return segmentation
    masks.

    :param image:

        Input Img.

    :return:

        Cellpose segmentation masks.
    """
    # create a temp directory
    tmp_dir = Files.createTempDirectory("fiji_tmp")

    # save input image
    tmp_save_path = tmp_dir.toString() + "/cellpose_input.tif"
    io.save(image, tmp_save_path)

    # build cellpose command
    if use_3d:
        cp_cmd = "cellpose --dir {} --pretrained_model {} --save_tif --do_3D --diam_mean {}".format(tmp_dir, cp_model, cp_diameter)
    else:
        cp_cmd = "cellpose --dir {} --pretrained_model {} --diameter {} --save_tif".format(tmp_dir, cp_model, cp_diameter)

    # run cellpose
    full_cmd = "{} -m {}".format(cp_py_path, cp_cmd)
    process = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    process.wait()

    # harvest cellpose results
    cp_masks_path = os.path.join(tmp_dir.toString(), "cellpose_input_cp_masks.tif")
    cp_masks = io.open(cp_masks_path)

    # remove temp directory
    shutil.rmtree(tmp_dir.toString())
    
    return cp_masks


def center_to_roi_manager(labeling):
    """Add the center of mass to the ROI manager.

    Add the center of mass for a label from a given ImgLabeling to
    the ROI Manager as a Point ROI.

    :param labeling:

        Input ImgLabeling
    """
    idx_img = labeling.getIndexImg()
    regs = LabelRegions(labeling)
    rm = RoiManager.getRoiManager()
    for r in regs:
        # get label ID
        r_id = Regions.sample(r, idx_img).firstElement()
        pos = r.getCenterOfMass().positionAsDoubleArray()
        roi = PointRoi(pos[0], pos[1])
        # set the Z position of the ROI
        roi.setName(str(r_id))
        roi.setPosition(0, int(round(pos[2])), 1)
        rm.addRoi(roi)


def extract_channel(image, channel, axis=2):
    """Extract a channel from the input image.

    Extract a channel from the input image.

    :param image:

        Input image.

    :param channel:

        String specifing which channel to extract.

    :param axis:

        The channel axis (default=2).

    :return:

        Extracted channel from the input image.
    """
    if channel == "None":
        return
    else:
        return ops.op("transform.hyperSliceView").input(image, axis, int(channel) - 1).apply()


def gaussian_high_pass_filter(image, sigma):
    """Apply a Gaussian high pass filter to an input image.

    Apply a Guassian high pass filter by applyinmg a guassian
    blur to the input image and subtracting from the original.

    :param image:

        Input image.

    :param sigma:

        Input Sigma value for Gaussian blur.

    :return:

        High pass filtered image.
    """
    gauss_image = ops.op("filter.gauss").input(image, sigma).apply()
    high_pass_image = ops.op("create.img").input(image, FloatType()).apply()
    ops.op("math.sub").input(image, gauss_image).output(high_pass_image).compute()

    return high_pass_image


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

    return cs.convert(idx_img, ImgLabeling)


def run(image):
    """Pipeline entry point.
    """
    # extract only specified channels
    chs = []
    chs.append(extract_channel(image, p_ch))
    chs.append(extract_channel(image, n_ch))

    # puncta: segment and add center of mass to the ROI Manager 
    pun_img_hp = gaussian_high_pass_filter(chs[0], hp_sigma)
    pun_img_th = ops.op("create.img").input(pun_img_hp, BitType()).apply()
    ops.op("threshold.{}".format(threshold)).input(pun_img_hp).output(pun_img_th).compute()
    pun_img_ws = ops.op("image.watershed").input(pun_img_th, False, False, ws_sigma, pun_img_th).apply()
    pun_img_ws = size_filter_labeling(pun_img_ws, min_size, max_size)
    center_to_roi_manager(pun_img_ws)
    if pun_show:
        ui.show("puncta label image", pun_img_ws.getIndexImg())

    # nuclear stain/marker - add to ROI Manager? 
    nuc_img_gb = ops.op("filter.gauss").input(chs[1], 2.0).apply()
    nuc_img_lb = cellpose(nuc_img_gb)
    if nuc_show:
        ui.show("nuclear label image", nuc_img_lb)

    # TODO add measurements

run(img)
