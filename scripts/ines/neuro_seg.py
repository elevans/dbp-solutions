import argparse
import imagej
import napari
import numpy as np
import code
from cellpose import models

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

def filter_labeling(index_img, labeling, min_size, max_size=None):
    """
    Filter an index image's labels by minimum/maximum pixel size exclusion.

    :param index_img: An index image.
    :param labeling: A labeling created from the index image.
    :param min_size: Minimum label size.
    :param max_size: Maximum label size (defaults row * col).
    """
    # use max image size to determine maximum size
    if max_size is None:
        max_size = index_img.shape[0] * index_img.shape[1]

    # get the label regions from the labeling
    regs = LabelRegions(labeling)
    for r in regs:
        # get a sample region and compute the size
        sample = Regions.sample(r, index_img)
        size = float(str(ij.op().stats().size(sample)))
        # if region size is outside of min/max range set to zero
        if size <= float(min_size) or size >= float(max_size):
            remove_label(sample)

def run_cellpose(image: np.ndarray, diameter: float, model: str) -> np.ndarray:
    """
    Run cellpose.
    """

    print(f"[STATUS]: Running Cellpose model {model}")
    model = models.CellposeModel(gpu=False, model_type=model)
    ch = [0, 0]
    results = model.eval(image, channels=ch, diameter=diameter)

    return results[0]

def segment_neurons(image: "net.imagej.Dataset") -> np.ndarray:
    """
    Segment the neurons
    """

    # convert to ImagePlus and use unsharp mask filter
    imp = ij.py.to_imageplus(image)
    ij.IJ.run(imp, "Unsharp Mask...", "radius=8 mask=0.75")

    # convert back to Img and threshold
    ds = ij.py.to_dataset(imp.duplicate())
    thres = ij.op().threshold().li(ds)

    # perform morphological operations
    thres = ij.op().morphology().dilate(thres, HyperShpereShape(5))

    # do CCA and filter small stuff
    results = ij.op().labeling().cca(thres, StructuringElement.EIGHT_CONNECTED)
    filter_labeling(results.getIndexImg(), results, 3000)

    return ij.py.to_xarray(results.getIndexImg()).data

# set up argpase
parser = argparse.ArgumentParser(description="Run segment comparison analysis")
parser.add_argument(
    "-i", "--input", required=True, help="Path to the image file."
)
pargs = parser.parse_args()

# initialize ImageJ
ij = imagej.init(mode='headless')
print(f"ImageJ2 version: {ij.getVersion()}")

# load extra ImageJ/Java resources
HyperShpereShape = imagej.sj.jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')
ImgLabeling = imagej.sj.jimport('net.imglib2.roi.labeling.ImgLabeling')
LabelRegions = imagej.sj.jimport('net.imglib2.roi.labeling.LabelRegions')
Regions = imagej.sj.jimport('net.imglib2.roi.Regions')
StructuringElement = imagej.sj.jimport('net.imglib2.algorithm.labeling.ConnectedComponents.StructuringElement')

# load data
img = ij.io().open(pargs.input)
xarr = ij.py.to_xarray(img)

# run analysis
cyto_labels = run_cellpose(xarr.data[:, :, 1], 30.0, "cyto2")
nuc_labels = run_cellpose(xarr.data[:, :, 0], 15.6, "CP")
neuro_labels = segment_neurons(img[:, :, 3])

# start napari viewer
viewer = napari.Viewer()
viewer.add_image(xarr.data, channel_axis=-1, name=str(img.getName()))
viewer.add_image(cyto_labels, name="cyto_labels", colormap="gist_earth")
viewer.add_image(nuc_labels, name="nuc_labels", colormap="gist_earth")
viewer.add_image(neuro_labels, name="neuro_labels")
# return the REPL
code.interact(local=locals())
