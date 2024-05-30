#@ OpEnvironment ops
#@ OpService ijops
#@ ConvertService cs
#@ IOService io
#@ UIService ui
#@ Img img
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

def compute_stats(image, labeling):
    """Compute geometry and other statics.
    """
    # extract regions
    regs = LabelRegions(labeling)
    reg_labels = regs.getExistingLabels()

    # set up table headers
    table = DefaultGenericTable(3, 0)
    table.setColumnHeader(0, "label")
    table.setColumnHeader(1, "size")
    table.setColumnHeader(2, "MFI")

    i = 0
    for r in regs:
        sample = Regions.sample(r, image)
        table.appendRow()
        table.set("label", i, int(r.getLabel()))
        table.set("size", i, ijops.stats().size(sample).getRealDouble())
        table.set("MFI", i, ijops.stats().mean(sample).getRealDouble())
        i += 1

    return table


def deconvolve(image):
    """Perform Richardson-Lucy deconvolution.
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
    """
    gauss = ops.op("filter.gauss").input(image, sigma).apply()
    
    gauss_sub = ops.op("create.img").input(image, FloatType()).apply()
    ops.op("math.sub").input(image, gauss).output(gauss_sub).compute()
    
    return gauss_sub


def run_cellpose(image):
    """Execute Cellpose via a conda environment.
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
    """
    views = []
    for i in range(image.dimensionsAsLongArray()[axis]):
        views.append(ops.op("transform.hyperSliceView").input(image, axis, i).apply())

    return views

def segment_nuclei(image):
    """Segment nuclei.
    """
    cp_index_img = run_cellpose(image)

    return cs.convert(cp_index_img, ImgLabeling)


def segment_puncta(image):
    """Segment puncta.
    """
    puncta_img = deconvolve(image)
    puncta_img = gaussian_subtraction(puncta_img, 9.0)

    thres = ops.op("create.img").input(puncta_img, BitType()).apply()
    ops.op("threshold.triangle").input(puncta_img).output(thres).compute()

    return ops.op("labeling.cca").input(thres, StructuringElement.EIGHT_CONNECTED).apply()

# split channels
chs = split_img(img)

# segment nuclei
nuc_img_labeling = segment_nuclei(chs[-1])

# segment puncta
puncta_img_labeling = segment_puncta(chs[1])

# show results
ui.show(nuc_img_labeling.getIndexImg())
ui.show(puncta_img_labeling.getIndexImg())

# show table
p_table = compute_stats(chs[1], puncta_img_labeling)
