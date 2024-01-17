import argparse
import imagej
import numpy as np
import napari
import code
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
res_cellpose = run_cellpose(xarr[:, :, :, -1].data)

# show output in napari
viewer = napari.Viewer()
viewer.add_image(xarr.data, channel_axis=3, name="input")
viewer.add_image(res_cellpose, name="cellpose_3d_output")

# return the REPL
code.interact(local=locals())