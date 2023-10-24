#@ ImgPlus img
#@ OpService ops
#@ ConvertService cs
#@ UIService ui
#@ IOService io
#@ Integer radius(label="FFT Radius:", value=0)
#@output ImgPlus output

from net.imglib2.img.display.imagej import ImageJFunctions
from net.imglib2.view import Views
from net.imglib2.util import Util
from net.imglib2.img import Img
from net.imglib2.algorithm.labeling.ConnectedComponents import StructuringElement
from net.imglib2.roi import Regions
from net.imglib2.roi.labeling import LabelRegions
from org.scijava.table import DefaultGenericTable

def fft_high_pass(image):
    """
    Processes 2D images.
    """
    # run FFT on the input image
    fft_img = ops.filter().fft(image)

    # get a cursor on the FFT array
    fft_cursor = fft_img.cursor()

    # note dimension order is X, Y or col, row
    cursor_pos = [None, None]
    origin_a = [0, 0]
    origin_b = [0, fft_img.dimension(1)]
    origin_c = [fft_img.dimension(0), 0]
    origin_d = [fft_img.dimension(0), fft_img.dimension(1)]

    # loop through the FFT array
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

    # reconstruct filtered image from FFT array and show
    recon = ops.create().img([image.dimension(0), image.dimension(1)])

    return ops.filter().ifft(recon, fft_img)

def split_channels(image, ch_axis=2):
    channels = []
    for i in range(image.dimensionsAsLongArray()[ch_axis]):
        channels.append(ops.transform().hyperSliceView(image, ch_axis, i))
    
    return channels

def segment_puncta(image, z_axis=2):
    # run 2D FFT on each slice at given radius
    fft_results = []
    for i in range(image.dimensionsAsLongArray()[z_axis]):
        s = ops.transform().hyperSliceView(image, z_axis, i)
        fft_results.append(fft_high_pass(s))

    # convert to Img/Dataset to threshold
    fft_img = cs.convert(Views.stack(fft_results), Img)
    thres = ops.threshold().triangle(fft_img)

    # run CCA analysis
    labeling = ops.labeling().cca(thres, StructuringElement.EIGHT_CONNECTED)

    return labeling

def compute_stats(labeling, image):
    # extract regions
    regs = LabelRegions(labeling)
    reg_labels = regs.getExistingLabels()

    # set up table
    table = DefaultGenericTable(2, len(reg_labels))
    table.setColumnHeader(0, "size")
    table.setColumnHeader(1, "MFI")

    for r in regs:
        sample = Regions.sample(r, image)
        table.set("size", int(r.getLabel()), ops.stats().size(sample).getRealDouble())
        table.set("MFI", int(r.getLabel()), ops.stats().mean(sample).getRealDouble())

    return table

# split channels
chs = split_channels(img)

# segment objects
puncta_labeling = segment_puncta(chs[1])

# compute stats
puncta_seg_results_table = compute_stats(puncta_labeling, chs[1])

# display results
ui.show(puncta_labeling.getIndexImg())
ui.show(puncta_seg_results_table)