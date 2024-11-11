#@ OpEnvironment ops
#@ OpService ijops
#@ ConvertService cs
#@ IOService io
#@ UIService ui
#@ Img img
#@ String (visibility = MESSAGE, value = "<b>Channel settings</b>", required = false) ch_msg
#@ Integer (label = "Nuclear channel position", min = 1, value = 1) ch_nuc
#@ Integer (label = "Puncta channel position", min = 1, value = 2) ch_pun
#@ Integer (label = "Nuclear marker channel position", min = 1, value = 3) ch_mrk
#@ String (visibility=MESSAGE, value="<b>(Nuclear Segmentation)  ---  Cellpose settings</b>", required=false) cp_msg
#@ String (label="Cellpose Python path:") cp_py_path 
#@ String (label="Pretrained model:", choices={"cyto", "cyto2", "nuclei"}, style="listBox", value = "nuclei") model
#@ Float (label="Diameter (2D only):", style="format:0.00", value=0.00, stepSize=0.01) diameter
#@ Boolean (label="Enable 3D segmentation:", value=False) use_3d
#@ String (visibility=MESSAGE, value="<b>(Puncta Segmentation)  ---  Deconvolution settings</b>", required=false) decon_msg
#@ Integer (label="Iterations:", value=15) iterations
#@ Float (label="Numerical Aperture:", style="format:0.00", min=0.00, value=1.40) numerical_aperture
#@ Integer (label="Emission Wavelength (nm):", value=461) wvlen
#@ Float (label="Refractive Index (immersion):", style="format:0.00", min=0.00, value=1.5) ri_immersion
#@ Float (label="Refractive Index (sample):", style="format:0.00", min=0.00, value=1.4) ri_sample
#@ Float (label="Lateral resolution (μm/pixel):", style="format:0.0000", min=0.0000, value=0.2205) lat_res
#@ Float (label="Axial resolution (μm/pixel):", style="format:0.0000", min=0.0000, value=0.26) ax_res
#@ Float (label="Particle/sample Position (μm):", style="format:0.0000", min=0.0000, value=0) pz
#@ Float (label="Regularization factor:", style="format:0.00000", min=0.00000, value=0.002) reg_factor
#@ String (visibility=MESSAGE, value="<b>(Puncta Segmentation)  ---  Threshold settings</b>", required=false) thresh_msg
#@ String (label="Global threshold method:", choices={"huang", "ij1", "intermodes", "isoData", "li", "maxEntropy", "maxLikelihood", "mean", "minError", "minimum", "moments", "otsu", "percentile", "renyiEntropy", "rosin", "shanbhag", "triangle", "yen"}, style="listBox", value = "triangle") thresh_method
#@ String (visibility=MESSAGE, value="<b>(Puncta Segmentation)  ---  Puncta size filter</b>", required=false) thresh_msg
#@ Integer (label = "Minimum puncta size", min = 0, value = 1) pun_min_size
#@ Integer (label = "Maximum puncta size", min = 0, value = 1) pun_max_size

import os
import subprocess
import shutil

from net.imglib2.img import Img
from net.imglib2.algorithm.labeling.ConnectedComponents import StructuringElement
from net.imglib2 import FinalDimensions
from net.imglib2.roi import Regions
from net.imglib2.roi.labeling import ImgLabeling, LabelRegions
from net.imglib2.type.logic import BitType
from net.imglib2.type.numeric.real import FloatType
from net.imglib2.type.numeric.complex import ComplexFloatType

from org.scijava.table import DefaultGenericTable

from java.nio.file import Files
from java.lang import Float

def check_for_null_mask(sample):
    """Check if a mask sample is null.

    Check if all the pixels within the given sample
    equal 0.

    :param sample:

        A labeling sample.

    :return:

        Boolean indicating whether all pixels within the
        given sample equal 0.
    """
    px_vals = list(set(sample_to_list(sample)))
    if len(px_vals) == 1 and px_vals[0] == 0:
        # all pixels are zero in sample
        return False
    else:
        return True


def sample_to_list(sample, skip=True):
    """Convert a sample's pixels into a list.

    :param sample:

        An ImgLib2 sample.

    :param skip:

        Skip the background label (i.e. skip 0).

    :return:

        A list of pixel values.
    """
    c = sample.cursor()
    vals = []
    while c.hasNext():
        c.fwd()
        v = c.get().getInteger()
        if skip and v == 0:
            continue
        else:
            vals.append(c.get().getInteger())

    return vals


def median(vals):
    """Compute the median from a list of integers.

    Compute the median value from a list of integers.

    :param vals:

        A list of integer values.

    :return:
        
        The median value of the list.
    """
    vals = sorted(vals)
    n = len(vals)
    mid = n // 2

    if n % 2 == 0:
        a = vals[mid -1]
        b = vals[mid]
        if a == b:
            return vals[mid]
        else:
            # reject double assignment of nuc
            return vals[0] # default to the lowest label number?
    else:
        return vals[mid]


def compute_puncta_stats(image, labeling):
    """Compute puncta geometry and other statistics.

    Compute and populate a table with puncta geometry and statistics.

    :param image:

        Input Img.

    :param labeling:

        An ImgLabeling.

    :return:

        A SciJava Table.
    """
    # extract regions
    regs = LabelRegions(labeling)

    # set up table headers
    table = DefaultGenericTable(3, 0)
    table.setColumnHeader(0, "label")
    table.setColumnHeader(1, "size (pixels)")
    table.setColumnHeader(2, "MFI")

    i = 0
    for r in regs:
        sample = Regions.sample(r, image)
        table.appendRow()
        table.set("label", i, int(r.getLabel()))
        table.set("size (pixels)", i, ijops.stats().size(sample).getRealDouble())
        table.set("MFI", i, ijops.stats().mean(sample).getRealDouble())
        i += 1

    return table


def compute_nuclear_stats(images, n_labeling, m_labeling, p_labeling):
    """Compute nuclear geometry and other statistics.

    Compute and populate a table with puncta geometry and statistics.

    :param n_labeling:

        Input nuclear ImgLabeling.

    :param m_labeling:

        Input marker ImgLabeling.

    :param p_labeling:

        Input puncta ImgLabeling.

    :return:

        A SciJava Table.
    """
    # extract regions
    n_regs = LabelRegions(n_labeling)
    p_regs = LabelRegions(p_labeling)
    p_ex_labels = p_regs.getExistingLabels()
    
    # set up table column headers
    table = DefaultGenericTable(4, 0)
    table.setColumnHeader(0, "nuclear label")
    table.setColumnHeader(1, "puncta labels")
    table.setColumnHeader(2, "number of puncta")
    table.setColumnHeader(3, "nuclear marker MFI")

    # get label/index images
    n_idx_img = n_labeling.getIndexImg()
    m_idx_img = m_labeling.getIndexImg()
    p_idx_img = p_labeling.getIndexImg()
   
    # create and populate results dicts/maps 
    n_p_map = {}
    n_m_map = {}
    for r in n_regs:
        # find all puncta labels within the nuclear ROI
        nuc_label = r.getLabel()
        nps = Regions.sample(r, p_idx_img)
        labels = list(set(sample_to_list(nps)))
        # compute the median label from the puncta sample
        median_labels = []
        for l in labels:
            # get label region, check against existing labels
            if l not in p_ex_labels:
                # skip if the label doesn't exist
                continue
            # create a sample of puncta region over nuclear labels
            pr = p_regs.getLabelRegion(l)
            ps = Regions.sample(pr, n_idx_img)
            # filter based on minimum and maximum bounds
            p_size = ops.op("stats.size").input(ps).apply()
            if p_size < pun_min_size or p_size > pun_max_size:
                # skip if the puncta size is outside min/max bounds
                continue
            # check for empty lists 
            pixel_vals = sample_to_list(ps)
            if not pixel_vals:
                continue
            else:
                # get median pixel value (i.e. which most of puncta signal exists)
                p_median = median(pixel_vals)
                if p_median == nuc_label:
                    # puncta is mostly in this mask, append
                    median_labels.append(l)
                else:
                    # puncta is mostly in another mask, skip
                    continue

        if median_labels:
            # create nuclear marker sample
            mks = Regions.sample(r, images[ch_mrk - 1])
            # populate the dictionaries
            n_p_map[nuc_label] = median_labels
            n_m_map[nuc_label] = ijops.stats().mean(mks).getRealDouble()
    
    i = 0
    for k, v in n_p_map.items():
        table.appendRow()
        table.set("nuclear label", i, k)
        table.set("puncta labels", i, v)
        table.set("number of puncta", i, len(v))
        table.set("nuclear marker MFI", i, n_m_map.get(k))
        i += 1

    return table


def count_sample_labels(sample):
    """Count the number of unique labels within a sample.

    Count the number of unique labels with a cursor within the
    given sample.

    :param sample:

        A labeling sample.

    :return:

        Number of unique labels within the given sample.
    """
    c = sample.cursor()
    out = []
    while c.hasNext():
        c.fwd()
        v = c.get().getInteger()
        if v != 0 and v not in out:
            out.append(v)
        else:
            continue

    return len(set(out))


def deconvolve(image):
    """Perform Richardson-Lucy TV deconvolution.

    Perform Richardson-Lucy Total Variation (TV) deconvolution
    on the input data.

    :param image:

        Input Img.

    :return:

        Deconvolved Img.
    """
    # convert parameters to meters
    wavelength = wvlen * 1E-9
    lateral_res = lat_res * 1E-6
    axial_res = ax_res * 1E-6
    z_pos = pz * 1E-6

    # convert input image to 32-bit
    image_f = ops.op("convert.float32").input(image).apply()
    
    # use image dimensions for PSF shape
    psf_shape = FinalDimensions(image_f.dimensionsAsLongArray())
    
    # create synthetic psf
    psf = ops.op("create.kernelDiffraction").input(psf_shape,
                                                   numerical_aperture,
                                                   wavelength,
                                                   ri_sample,
                                                   ri_immersion,
                                                   lateral_res,
                                                   axial_res,
                                                   z_pos,
                                                   FloatType()).apply()
    
    decon_img= ops.op("deconvolve.richardsonLucyTV").input(image_f,
                                                       psf,
                                                       FloatType(),
                                                       ComplexFloatType(),
                                                       iterations,
                                                       False,
                                                       False,
                                                       Float(reg_factor)).apply()
    return decon_img


def gaussian_subtraction(image, sigma):
    """Perform a Gaussian subraction.

    Apply a Gaussian blur on the input image at the given
    sigma value and subtract it from the input.

    :param image:

        Input Img.

    :param sigma:

        Gaussian blur sigma value.

    :return:

        Gaussian blur subtracted Img.
    """
    gauss = ops.op("filter.gauss").input(image, sigma).apply()
    
    gauss_sub = ops.op("create.img").input(image, FloatType()).apply()
    ops.op("math.sub").input(image, gauss).output(gauss_sub).compute()
    
    return gauss_sub


def run_cellpose(image):
    """Execute Cellpose via a conda environment.

    Run cellpose with the given settings via a conda environment.

    :param image:

        Input Img.

    :return:

        Cellpose output.
    """
    # create a temp directory
    tmp_dir = Files.createTempDirectory("fiji_tmp")

    # save input image
    tmp_save_path = tmp_dir.toString() + "/cellpose_input.tif"
    io.save(image, tmp_save_path)

    # build cellpose command
    if use_3d:
        cp_cmd = "cellpose --dir {} --pretrained_model {} --save_tif --do_3D".format(tmp_dir, model)
    else:
        cp_cmd = "cellpose --dir {} --pretrained_model {} --diameter {} --save_tif".format(tmp_dir, model, diameter)

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


def run_cellpose_labeling(image):
    """Run cellpose and convert to ImgLabeling.

    Run cellpose and return an ImgLabeling.

    :param image:

        Input Img.

    :return:

        Output cellpose ImgLabeling.
    """
    cp_index_img = run_cellpose(image)

    return cs.convert(cp_index_img, ImgLabeling)


def run_puncta_labeling(image):
    """Run puncta image processing steps.

    Run the puncta image processing steps and return
    an ImgLabeling.

    :param image:

        Input Img.

    :return:

        Output puncta ImgLabeling.
    """
    print("[INFO]: Deconvolving image...")
    puncta_img = deconvolve(image)
    print("[INFO]: Applying Gaussian Subtraction...")
    puncta_img = gaussian_subtraction(puncta_img, 9.0)

    print("[INFO]: Creating image labeling...")
    thres = ops.op("create.img").input(puncta_img, BitType()).apply()
    ops.op("threshold.{}".format(thresh_method)).input(puncta_img).output(thres).compute()

    return ops.op("labeling.cca").input(thres, StructuringElement.EIGHT_CONNECTED).apply()


# TODO:
# Wants
# - Link to the ROI manager (link table to image)
# - Better seperate the puncta in Z axis
# - Investigate OSError on macOS analysis computer
# - Need a better way to decide puncta nuclear assignment (1)
#   - Where the max signal is? Use that to assign to nucleus
#   - Where the centroid mass is? Use that to assign to nucleus
#   - Where sum of the signal (area sum) of the puncta?
# - Filter for puncta size before nuclear assignment (1)
#   - Allow pixel size entry
#   - Double nuclear rejection (i.e. if a nucleus has a doublet
#     remove that nucleus and all associated puncta from the analysis
#   - All of this should be optional
# - Better geen channel measurements (klf4l) (3)
#   - Problem is the pixel MFI is kinda weak -- can we do better?
#   - Crop based on nuclear bounding box, threshold the green signal
#     check for threshold size (signal should be within some adjustable
#     threshold for the ratio of nuclear size).
# - Support for dim puncta (2)
#   - maybe add optinal boolean for dim puncta extra steps
#   - Yu-Hsuan will send data
# split channels
chs = split_img(img)

# segment data
print("[INFO]: Running Cellpose on nuclei...")
nuc_img_labeling = run_cellpose_labeling(chs[ch_nuc - 1])
print("[INFO]: Running Cellpose on nuclear marker...")
mrk_img_labeling = run_cellpose_labeling(chs[ch_mrk - 1])
print("[INFO]: Running puncta segmentation...")
pun_img_labeling = run_puncta_labeling(chs[ch_pun - 1])

# show results
ui.show("nuclear labeling", nuc_img_labeling.getIndexImg())
ui.show("puncta labeling", pun_img_labeling.getIndexImg())
ui.show("nuclear marker labeling", mrk_img_labeling.getIndexImg())

# show table
print("[INFO]: Computing nuclear stats...")
n_table = compute_nuclear_stats(chs, nuc_img_labeling, mrk_img_labeling, pun_img_labeling)
print("[INFO]: Computing puncta stats...")
p_table = compute_puncta_stats(chs[1], pun_img_labeling)
ui.show("nuclear results table", n_table)
ui.show("puncta results table", p_table)
