#@ OpService ops
#@ UIService ui
#@ Img (label = "Input image:", autofill = false) img
#@ String (visibility = MESSAGE, value ="<b>[ RLTV deconvolution settings ]</b>", required = false) decon_msg
#@ Integer iterations(label="Iterations", value=15)
#@ Float (label="Numerical Aperture", style="format:0.00", min=0.00, value=1.45) na
#@ Integer (label="Emission Wavelength (nm)", value=457) wavelength
#@ Float (label="Refractive Index (immersion)", style="format:0.00", min=0.00, value=1.5) ri_immersion
#@ Float (label="Refractive Index (sample)", style="format:0.00", min=0.00, value=1.4) ri_sample
#@ Float (label="XY axis pixel spacing (um/pixel)", style="format:0.0000", min=0.0000, value=0.065) xy_res
#@ Float (label="Z axis pixel spacing (um/pixel)", style="format:0.0000", min=0.0000, value=0.1) z_res
#@ Float (label="Particle/sample Position (um)", style="format:0.0000", min=0.0000, value=0) p_z
#@ Float (label="Regularization factor", style="format:0.00000", min=0.00000, value=0.002) reg_factor
#@ String (visibility = MESSAGE, value ="<b>[ Background suppression settings ]</b>", required = false) bk_msg
#@ Integer (label = "Mean filter radius:", min = 0, value = 6) radius
#@ String (choices={"Diamond", "Hyper Sphere"}, style="listBox", value = "Hyper Sphere") shape
#@ Float (label="Gaussian blur Sigma:", style="format:0.00", min=0.0, value=7.00) sigma
#@output Img result

from net.imglib2 import FinalDimensions
from net.imglib2.algorithm.neighborhood import (DiamondShape,
                                                HyperSphereShape)
from net.imglib2.type.numeric.real import FloatType

shape_map = {
    "Diamond": DiamondShape,
    "Hyper Sphere": HyperSphereShape,
}

def rltv_deconvolution(image, na, wavelength, ri_sample, ri_immersion, xy_res, z_res, p_z, iterations, reg_factor):
    """Deconvolve an image with the Richardson-Lucy Total Variation (RTLV) algorithm.  
    """
    image = ops.convert().float32(image)
    psf_size = FinalDimensions(image.dimensionsAsLongArray())
    wavelength = float(wavelength) * 1E-9
    xy_res = xy_res * 1E-6
    z_res = z_res * 1E-6
    p_z = p_z * 1E-6
    psf = ops.create().kernelDiffraction(
        psf_size,
        na,
        wavelength,
        ri_sample,
        ri_immersion,
        xy_res,
        z_res,
        p_z,
        FloatType()
        )

    return ops.deconvolve().richardsonLucyTV(image, psf, iterations, reg_factor)


def suppress_background(image):
    # apply mean filter
    image = ops.convert().float32(image)
    image_mean = ops.create().img(image)
    ops.filter().mean(image_mean, img, shape_map.get(shape)(radius))
    image = ops.math().multiply(image, image_mean)
    # apply high pass Gaussian filter
    image_gauss = ops.filter().gauss(image, sigma)

    return ops.math().subtract(image, image_gauss)

# TODO: Results table: two tables -> per cell ID table (ID, area, diameter, sphereoscity, MFI) and summary (total cells found, largest found (size), smallest found (size)) tablei
# TODO: Results table 2D vs 3D title
# TODO: Add size filter with min and maximum bounds

decon = rltv_deconvolution(img, na, wavelength, ri_sample, ri_immersion, xy_res, z_res, p_z, iterations, reg_factor)
result = suppress_background(decon)
