import imagej
import imagej.convert as convert
import numpy as np
import scyjava as sj
import code
from imagej._java import jc
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from cellpose import models

def run_stardist(image: np.ndarray) -> np.ndarray:
    """Run 2D StarDist on an input image.
    """
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    nuc_labels, _ = model.predict_instances(normalize(image))

    return nuc_labels


def run_cellpose(image: np.ndarray) -> np.ndarray:
    """Run 2D cellpose on an input image
    """
    # run Cellpose on cytoplasm (grayscale)
    model = models.CellposeModel(gpu=False, model_type='cyto')
    ch = [0, 0]
    cyto_labels = model.eval(image, channels=ch, diameter=72.1)

    return cyto_labels[0]


def segment_puncta(image: "jc.RandomAccessibleInterval", radius: int):
    Util = sj.jimport('net.imglib2.util.Util')

    # run FFT on the input image
    img = ij.convert().convert(image, jc.Img)
    breakpoint()
    image_fft = ij.op().filter().fft(img)

    # setup curor and origins
    fft_cursor = image_fft.cursor()
    cursor_pos = [None, None]
    origin_a = [0, 0]
    origin_b = [0, image_fft.shape[1]]
    origin_c = [image_fft.shape[0], 0]
    origin_d = [image_fft.shape[0], image_fft.shape[1]]

    # filter the frequency domain image
    while fft_cursor.hasNext():
        # advance cursor and localize
        fft_cursor.fwd()
        cursor_pos[0] = fft_cursor.getLongPosition(0)
        cursor_pos[1] = fft_cursor.getLongPosition(1)

        # calculate distance from origins
        dist_a = Util.distance(origin_a, cursor_pos)
        dist_b = Util.distance(origin_b, cursor_pos)
        dist_c = Util.distance(origin_c, cursor_pos)
        dist_d = Util.distance(origin_d, cursor_pos)
        
        # Remove low frequences
        if (dist_a <= radius) or \
            (dist_b <= radius) or \
            (dist_c <= radius) or \
            (dist_d <= radius):
            fft_cursor.get().setZero()

    # reconstruct the filtered image
    CreateNamespace = sj.jimport('net.imagej.ops.create.CreateNamespace')
    recon = ij.op().namespace(CreateNamespace).img([img.shape[0], img.shape[1]])
    ij.op().filter().ifft(recon, image_fft)

    return recon


# initialize imagej
ij = imagej.init(mode='interactive')
dataset = ij.io().open("/home/edward/Documents/workspaces/coba/dbps/newmark/data/20230320_6_sm_Spol_aadc_klf4(633)_piwi1(G)_telo(R)_tes_63x_test.tif")

# process the channels
nuc_labels = run_stardist(ij.py.from_java(dataset[:, :, -1, 11])) # lets use slice 11 for now
puncta = segment_puncta(dataset[:, :, 1, 11], 100)

# convert index images to ImageJ ROIs
convert.index_img_to_roi_manager(ij, nuc_labels)

# display the original dataset
ij.ui().show(dataset)
ij.ui().show(puncta)

# return the REPL
code.interact(local=locals())