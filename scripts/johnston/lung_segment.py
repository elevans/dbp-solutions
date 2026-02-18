#@ OpEnvironment ops
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
#@ String (visibility = MESSAGE, value ="<b>[ Segmentation settings ]</b>", required = false) seg_msg
#@ String (label="Global threshold method:", choices={"huang", "ij1", "intermodes", "isoData", "li", "maxEntropy", "maxLikelihood", "mean", "minError", "minimum", "moments", "otsu", "percentile", "renyiEntropy", "rosin", "shanbhag", "triangle", "yen"}, style="listBox") method
#@ Integer min_size(label="Minimum size (pixels):", min=0, value=0)
#@ Integer max_size(label="Maximum size (pixels):", value=0)
#@ String (visibility = MESSAGE, value ="<b>[ Data pre-processing settings ]</b>", required = false) pp_msg
#@ Boolean (label = "RLTV deconvolution:", value = true) do_rltv
#@ Boolean (label = "Suppress background:", value = true) do_bksp
#@output Img result

from net.imglib2 import FinalDimensions
from net.imglib2.algorithm.labeling.ConnectedComponents import StructuringElement
from net.imglib2.algorithm.neighborhood import HyperSphereShape
from net.imglib2.roi.labeling import ImgLabeling, LabelRegions
from net.imglib2.roi import Regions
from net.imglib2.type.logic import BitType
from net.imglib2.type.numeric.real import FloatType
from net.imglib2.type.numeric.complex import ComplexFloatType
from net.imglib2.view import Views
from java.lang import Float

def filter_labeling(index_img, labeling, min_size, max_size):
    """Filter an index image's labels by minimum/maximum pixel size exclusion.
    """
    # get the label regions from the labeling
    regs = LabelRegions(labeling)
    for r in regs:
        # get a sample region and compute the size
        sample = Regions.sample(r, index_img)
        size = float(ops.op("stats.size").input(sample).apply())
        # if region size is outside of min/max range set to zero
        if size <= float(min_size) or size >= float(max_size):
            remove_label(sample)

def rltv_deconvolution(image, na, wavelength, ri_sample, ri_immersion, xy_res, z_res, p_z, iterations, reg_factor):
    """Deconvolve an image with the Richardson-Lucy Total Variation (RTLV) algorithm.  
    """
    psf_size = FinalDimensions(image.dimensionsAsLongArray())
    wavelength = float(wavelength) * 1E-9
    xy_res = xy_res * 1E-6
    z_res = z_res * 1E-6
    p_z = p_z * 1E-6
    psf = ops.op("create.kernelDiffraction").input(psf_size,
                                                   na,
                                                   wavelength,
                                                   ri_sample,
                                                   ri_immersion,
                                                   xy_res,
                                                   z_res,
                                                   p_z,
                                                   FloatType()
                                               ).apply()

    return ops.op("deconvolve.richardsonLucyTV").input(image,
                                                       psf,
                                                       FloatType(),
                                                       ComplexFloatType(),
                                                       iterations,
                                                       False,
                                                       False,
                                                       Float(reg_factor)).apply()


def remove_label(sample):
    """Set the given sample region pixel values to 0.
    """
    c = sample.cursor()
    while c.hasNext():
        c.fwd()
        c.get().set(0)


def suppress_background(image):
    """Supress the background with a mean and Gaussian high pass filter.  
    """
    # apply mean filter
    shape = HyperSphereShape(radius)
    if mean_filter_dim == "3D":
        img_mean = ops.op("create.img").input(image, FloatType()).apply()
        ops.op("filter.mean").input(image, shape).output(img_mean).compute()
    else:
        stack = []
        for i in range(image.dimensionsAsLongArray()[2]):
            view = ops.op("transform.hyperSliceView").input(img, 2, i).apply()
            m_img = ops.op("create.img").input(view).apply()
            ops.op("filter.mean").input(view, shape).output(m_img).compute()
            stack.append(ops.op("transform.addDimensionView").input(m_img, 1, 1).apply())
        img_mean = Views.concatenate(2, stack)
    # multiply the mean with the input image
    img_mul = ops.op("create.img").input(image, FloatType()).apply()
    img_blur = ops.op("create.img").input(image, FloatType()).apply()
    ops.op("math.multiply").input(image, img_mean).output(img_mul).compute()
    # apply high pass Gaussian filter
    img_blur = ops.op("filter.gauss").input(img_mul, sigma).apply()

    return ops.op("math.subtract").input(img_mul, img_blur).apply()


def to_f32(image):
    """Convert the input image to float32.
    """
    img_float = ops.op("create.img").input(img, FloatType()).apply()
    ops.op("convert.float32").input(image).output(img_float).compute()

    return img_float


def segment(image):
    """Use simple threshold and CCA to create label image segmentation.
    """
    # apply threshold
    mask = ops.op("create.img").input(image, BitType()).apply()
    ops.op("threshold.{}".format(method)).input(image).output(mask).compute()
    labeling = ops.op("labeling.cca").input(mask, StructuringElement.FOUR_CONNECTED).apply()
    
    return labeling

# TODO: Results table: two tables -> per cell ID table (ID, area, diameter, sphereoscity, MFI) and summary (total cells found, largest found (size), smallest found (size)) tablei
# TODO: Results table 2D vs 3D title

image = to_f32(img)
if do_rltv:
    image = rltv_deconvolution(image, na, wavelength, ri_sample, ri_immersion, xy_res, z_res, p_z, iterations, reg_factor)
if do_bksp:
    image = suppress_background(image)
labeling = segment(image)
filter_labeling(labeling.getIndexImg(), labeling, min_size, max_size)
result = labeling.getIndexImg()

