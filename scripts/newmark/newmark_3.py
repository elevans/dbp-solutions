#@ OpEnvironment ops
#@ OpService ijops
#@ ConvertService cs
#@ IOService io
#@ UIService ui
#@ Img img
#@ String (visibility=MESSAGE, value="<b>Channel settings</b>", required=false) ch_msg
#@ Integer (label = "Nuclear channel position", min = 1, value = 1) ch_nuc
#@ Integer (label = "Puncta channel position", min = 1, value = 2) ch_pun
#@ Integer (label = "Nuclear marker channel position", min = 1, value = 3) ch_mrk
#@ String (visibility=MESSAGE, value="<b>Cellpose settings</b>", required=false) cp_msg
#@ String (label="Cellpose Python path:") cp_py_path 
#@ String (label="Pretrained model:", choices={"cyto", "cyto2", "nuclei"}, style="listBox") model
#@ Float (label="Diameter (2D only):", style="format:0.00", value=0.00, stepSize=0.01) diameter
#@ Boolean (label="Enable 3D segmentation:", value=False) use_3d
#@ String (visibility=MESSAGE, value="<b>Deconvolution settings</b>", required=false) decon_msg
#@ Integer (label="Iterations:", value=15) iterations
#@ Float (label="Numerical Aperture:", style="format:0.00", min=0.00, value=1.40) numerical_aperture
#@ Integer (label="Emission Wavelength (nm):", value=461) wvlen
#@ Float (label="Refractive Index (immersion):", style="format:0.00", min=0.00, value=1.5) ri_immersion
#@ Float (label="Refractive Index (sample):", style="format:0.00", min=0.00, value=1.4) ri_sample
#@ Float (label="Lateral resolution (μm/pixel):", style="format:0.0000", min=0.0000, value=0.2205) lat_res
#@ Float (label="Axial resolution (μm/pixel):", style="format:0.0000", min=0.0000, value=0.26) ax_res
#@ Float (label="Particle/sample Position (μm):", style="format:0.0000", min=0.0000, value=0) pz
#@ Float (label="Regularization factor:", style="format:0.00000", min=0.00000, value=0.002) reg_factor

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
    c = sample.cursor()
    px_vals = []
    while c.hasNext():
        c.fwd()
        px_vals.append(c.get().getInteger())
    px_vals = list(set(px_vals))
    if len(px_vals) == 1 and px_vals[0] == 0:
        # all pixels are zero in sample
        return True
    else:
        return False


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
    reg_labels = regs.getExistingLabels()

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


def compute_nuclear_stats(n_labeling, m_labeling, p_labeling):
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
    # extract nuclear regions
    n_regs = LabelRegions(n_labeling)
    n_regs_labels = n_regs.getExistingLabels()

    # set up table column headers
    table = DefaultGenericTable(4, 0)
    table.setColumnHeader(0, "label")
    table.setColumnHeader(1, "size (pixels)")
    table.setColumnHeader(2, "marker status")
    table.setColumnHeader(3, "number of puncta")

    i = 0
    m_idx_img = m_labeling.getIndexImg()
    p_idx_img = p_labeling.getIndexImg()
    for r in n_regs:
        ms = Regions.sample(r, m_idx_img)
        ps = Regions.sample(r, p_idx_img)
        table.appendRow()
        table.set("label", i , int(r.getLabel()))
        table.set("size (pixels)", i, ops.op("stats.size").input(ps).apply())
        mrk_status = check_for_null_mask(ms)
        if mrk_status is True:
            table.set("marker status", i, 1)
        else:
            table.set("marker status", i, 0)
        table.set("number of puncta", i, count_sample_labels(ps))
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
    ops.op("threshold.triangle").input(puncta_img).output(thres).compute()

    return ops.op("labeling.cca").input(thres, StructuringElement.EIGHT_CONNECTED).apply()


# TODO:
# Wants
# - Link to the ROI manager (link table to image)

# split channels
chs = split_img(img)

# segment data
nuc_img_labeling = run_cellpose_labeling(chs[ch_nuc - 1])
mrk_img_labeling = run_cellpose_labeling(chs[ch_mrk - 1])
pun_img_labeling = run_puncta_labeling(chs[ch_pun - 1])

# show results
ui.show("nuclear labeling", nuc_img_labeling.getIndexImg())
ui.show("puncta labeling", pun_img_labeling.getIndexImg())
ui.show("nuclear marker labeling", mrk_img_labeling.getIndexImg())

# show table
n_table = compute_nuclear_stats(nuc_img_labeling, mrk_img_labeling, pun_img_labeling)
p_table = compute_puncta_stats(chs[1], pun_img_labeling)
ui.show("nuclear results table", n_table)
ui.show("puncta results table", p_table)
