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
    print("[STATUS]: Running Cellpose 3D...")
    model = models.CellposeModel(gpu=False, model_type="nuclei")
    ch = [0, 0]
    results = model.eval(image, channels=ch, diameter=40, do_3D=True, z_axis=0)
    
    return results[0]

def run_fft_high_pass(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Filter with a high-pass filter.
    Allows edges, details and noise to pass through while attenuating the low-
    frequency components (e.g. smooth variations and background)
    """
    print("[STATUS]: Running FFT ...")
    # compute FFT on input image
    fft_narr = np.fft.fftn(image)
    fft_shift_narr = np.fft.fftshift(fft_narr)

    # create circular high pass filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.uint8)

    # set the center of the mask to zero (block the DC component)
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

    # perform high pass filter
    shift_high_pass_result = fft_shift_narr * mask

    # shift the zero-frequency components back to the corners
    high_pass_result = np.fft.ifftshift(shift_high_pass_result)

    # reconstruct the filtered image from the array
    recon_narr = np.fft.ifftn(high_pass_result)

    return np.real(recon_narr)


def segment_nuclei(image: "net.imagej.ImgPlus") -> np.ndarray:
    res_cellpose = run_cellpose(ij.py.to_xarray(image[:, :, -1, :]).data)

    return res_cellpose


def segment_puncta(image: xr.DataArray, radius:int, min_size:int, max_size:int, z_axis=0) -> np.ndarray:
    """Segment puncta using FFT.
    """
    fft_results = []
    for i in range(image.shape[z_axis]):
        s = image[i, :, :, 1].data
        fft_results.append(ij.py.to_java(run_fft_high_pass(s, radius)))

    # convert the ImgPlus
    fft_img = ij.convert().convert(Views.stack(fft_results), Img)
    thres = ij.op().threshold().triangle(fft_img)

    # run CCA
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
    "-r", "--radius", type=int, required=True, help="Radius value as an integer."
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
Views = sj.jimport('net.imglib2.view.Views')
StructuringElement = sj.jimport('net.imglib2.algorithm.labeling.ConnectedComponents.StructuringElement')
ImgLabeling = sj.jimport('net.imglib2.roi.labeling.ImgLabeling')
LabelRegions = sj.jimport('net.imglib2.roi.labeling.LabelRegions')
Regions = sj.jimport('net.imglib2.roi.Regions')

# load data
img = ij.io().open(pargs.input)
xarr = ij.py.to_xarray(img)

# run analysis
nuclei_labels = segment_nuclei(img)
puncta_labels = segment_puncta(ij.py.to_xarray(img), pargs.radius, pargs.min_size, pargs.max_size)

# show output in napari
viewer = napari.Viewer()
viewer.add_image(xarr.data, channel_axis=3, name="input")
viewer.add_image(nuclei_labels, name="nuclei_labels")
viewer.add_image(puncta_labels, name="puncta_labels")

# return the REPL
code.interact(local=locals())