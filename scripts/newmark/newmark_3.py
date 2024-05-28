#@ OpEnvironment ops
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

from net.imglib2 import FinalDimensions
from net.imglib2.type.numeric.real import FloatType
from net.imglib2.type.numeric.complex import ComplexFloatType

from java.nio.file import Files
from java.lang import Float

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

# split channels
chs = split_img(img)

# segment nuclei
cp_masks = run_cellpose(chs[-1])

# segment puncta
decon = deconvolve(chs[1])
puncta = gaussian_subtraction(decon, 9.0)

# show results
ui.show(cp_masks)
ui.show(puncta)
