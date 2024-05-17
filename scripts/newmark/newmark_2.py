import argparse
import imagej
import scyjava as sj
import numpy as np
import napari
import code
import xarray as xr
from cellpose import models

def run_cellpose(image: np.ndarray) -> np.ndarray:
    """
    Run Cellpose.
    """
    print("[INFO]: Running Cellpose 3D...")
    model = models.CellposeModel(gpu=False, model_type="nuclei")
    ch = [0, 0]
    results = model.eval(image, channels=ch, diameter=40, do_3D=True, z_axis=0)
    
    return results[0]


def segment_nuclei(image: "net.imagej.ImgPlus") -> np.ndarray:
    res_cellpose = run_cellpose(ij.py.to_xarray(image[:, :, -1, :]).data)

    return res_cellpose


def segment_puncta(image: "net.imagej.ImgPlus", min_size:int, max_size:int, z_axis=0) -> np.ndarray:
    """Segment puncta using deconvolution and gaussian blur subtraction.

    :param image: Input image
    :param min_size: Minimum object size
    :param max_Size: Maximum object size
    :param z_axis: Z-axis position (default=0)
    :return: Segmented puncta
    """
    # convert to float
    image = ij.op().convert().float32(image)

    # deconvolve data
    na = 1.4
    ri_immersion = 1.5
    ri_sample = 1.4
    wavelength = 461 * 1E-9
    lateral_res = 0.2205 * 1E-6
    axial_res = 0.26 * 1E-6
    pZ = 0
    size = FinalDimensions(image[:, :, 1, :].dimensionsAsLongArray())
    create = ij.op().namespace(CreateNamespace)
    psf = create.kernelDiffraction(size, na, wavelength, ri_sample, ri_immersion, lateral_res, axial_res, pZ, FloatType())
    print("[INFO]: Deconvolving data...")
    puncta_img = ij.op().deconvolve().richardsonLucyTV(image[:, :, 1, :], psf, 15, 0.002)

    # perform gaussian subtraction
    puncta_img_gauss = ij.op().filter().gauss(puncta_img, 9.0)
    puncta_img_sub = puncta_img - puncta_img_gauss

    # run CCA
    thres = ij.op().threshold().triangle(puncta_img_sub)
    res_puncta = ij.op().labeling().cca(thres, StructuringElement.EIGHT_CONNECTED)

    # filter puncta
    filter_labeling(res_puncta.getIndexImg(), res_puncta, min_size, max_size)

    return ij.py.to_xarray(res_puncta.getIndexImg())


def remove_label(sample):
    """
    Set the given sample region pixel values to 0.

    :param: A sample region.
    """
    # get the sample region's cursor
    c = sample.cursor()
    # set all pixels within the sample region to 0
    while c.hasNext():
        c.fwd()
        c.get().set(0)


def filter_labeling(index_img, labeling, min_size, max_size):
    """
    Filter an index image's labels by minimum/maximum pixel size exclusion.

    :param index_img: An index image.
    :param labeling: A labeling created from the index image.
    :param min_size: Minimum label size.
    :param max_size: Maximum label size.
    """
    # get the label regions from the labeling
    regs = LabelRegions(labeling)
    for r in regs:
        # get a sample region and compute the size
        sample = Regions.sample(r, index_img)
        size = float(str(ij.op().stats().size(sample)))
        # if region size is outside of min/max range set to zero
        if size <= float(min_size) or size >= float(max_size):
            remove_label(sample)

# set up argpase
parser = argparse.ArgumentParser(description="Run segment comparison analysis")
parser.add_argument(
    "-i", "--input", required=True, help="Path to the image file."
)
parser.add_argument(
    "--min_size", type=int, required=True, help="Minimum size value as an integer."
)
parser.add_argument(
    "--max_size", type=int, required=True, help="Maximum size value as an integer."
)
pargs = parser.parse_args()

# initialize ImageJ
ij = imagej.init(mode='headless')
print(f"ImageJ2 version: {ij.getVersion()}")

# import extra ImageJ2/Imglib2 resources
Util = sj.jimport('net.imglib2.util.Util')
Img = sj.jimport('net.imglib2.img.Img')
FinalDimensions = sj.jimport('net.imglib2.FinalDimensions')
Views = sj.jimport('net.imglib2.view.Views')
StructuringElement = sj.jimport('net.imglib2.algorithm.labeling.ConnectedComponents.StructuringElement')
ImgLabeling = sj.jimport('net.imglib2.roi.labeling.ImgLabeling')
LabelRegions = sj.jimport('net.imglib2.roi.labeling.LabelRegions')
Regions = sj.jimport('net.imglib2.roi.Regions')
CreateNamespace = sj.jimport('net.imagej.ops.create.CreateNamespace')
FloatType = sj.jimport('net.imglib2.type.numeric.real.FloatType')

# load data
img = ij.io().open(pargs.input)
xarr = ij.py.to_xarray(img)

# run analysis
nuclei_labels = segment_nuclei(img)
puncta_labels = segment_puncta(img, pargs.min_size, pargs.max_size)

# show output in napari
viewer = napari.Viewer()
viewer.add_image(xarr.data, channel_axis=3, name="input")
viewer.add_image(nuclei_labels, name="nuclei_labels")
viewer.add_image(puncta_labels, name="puncta_labels")

# return the REPL
code.interact(local=locals())
