#@ OpService ops
#@ UIService ui
#@ Img (label = "Input image:", autofill = false) img
#@ String (visibility = MESSAGE, value ="<b>[ Richardson-Lucy Total Variation (RLTV) deconvolution settings ]</b>", required = false) decon_msg
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
#@ String (label = "Mean filter dimensonality:", choices={"2D", "3D"}, style="listBox", value = "3D") mean_filter_dim
#@ Float (label="Gaussian blur Sigma:", style="format:0.00", min=0.0, value=25.00) sigma
#@ String (visibility = MESSAGE, value ="<b>[ Data pre-processing settings ]</b>", required = false) pp_msg
#@ Boolean (label = "RLTV deconvolution:", value = true) do_rltv
#@ Boolean (label = "Suppress background:", value = true) do_bksp
#@output Img result

from net.imglib2 import FinalDimensions
from net.imglib2.algorithm.neighborhood import HyperSphereShape
from net.imglib2.type.numeric.real import FloatType
from net.imglib2.view import Views

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
    if mean_filter_dim == "3D":
        image_mean = ops.create().img(image)
        ops.filter().mean(image_mean, img, HyperSphereShape(radius))
    else:
        shape = HyperSphereShape(radius)
        stack = []
        for i in range(image.dimensionsAsLongArray()[2]):
            view = ops.transform().hyperSliceView(image, 2, i)
            m_img = ops.create().img(view)
            ops.filter().mean(m_img, view, shape)
            stack.append(ops.transform().addDimensionView(m_img, 1, 1))
        image_mean = Views.concatenate(2, stack)
    image = ops.math().multiply(image, image_mean)
    # apply high pass Gaussian filter
    image_gauss = ops.filter().gauss(image, sigma)

    return ops.math().subtract(image, image_gauss)

# TODO: Results table: two tables -> per cell ID table (ID, area, diameter, sphereoscity, MFI) and summary (total cells found, largest found (size), smallest found (size)) tablei
# TODO: Results table 2D vs 3D title
# TODO: Add size filter with min and maximum bounds

if do_rltv:
    img = rltv_deconvolution(img, na, wavelength, ri_sample, ri_immersion, xy_res, z_res, p_z, iterations, reg_factor)
if do_bksp:
    img = suppress_background(img)

result = img
