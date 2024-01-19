import argparse
import imagej
import napari
import numpy as np
import code
from cellpose import models

def run_cellpose(image: np.ndarray, diameter: float, model: str) -> np.ndarray:
    """
    Run cellpose.
    """

    print(f"[STATUS]: Running Cellpose model {model}")
    model = models.CellposeModel(gpu=False, model_type=model)
    ch = [0, 0]
    results = model.eval(image, channels=ch, diameter=diameter)

    return results[0]

# set up argpase
parser = argparse.ArgumentParser(description="Run segment comparison analysis")
parser.add_argument(
    "-i", "--input", required=True, help="Path to the image file."
)
pargs = parser.parse_args()

# initialize ImageJ
ij = imagej.init(mode='headless')
print(f"ImageJ2 version: {ij.getVersion()}")

# load data
img = ij.io().open(pargs.input)
xarr = ij.py.to_xarray(img)

# run analysis
cyto_labels = run_cellpose(xarr.data[:, :, 1], 30.0, "cyto2")
nuc_labels = run_cellpose(xarr.data[:, :, 0], 15.6, "CP")

# start napari viewer
viewer = napari.Viewer()
viewer.add_image(xarr.data, channel_axis=-1, name=str(img.getName()))
viewer.add_image(cyto_labels, name="cyto_labels", colormap="gist_earth")
viewer.add_image(nuc_labels, name="nuc_labels", colormap="gist_earth")

# return the REPL
code.interact(local=locals())