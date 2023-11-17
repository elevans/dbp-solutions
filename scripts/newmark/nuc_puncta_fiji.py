#@ ImgPlus img
#@ OpService ops
#@ CommandService cmd
#@ ConvertService cs
#@ UIService ui
#@ IOService io
#@ Integer radius(label="FFT Radius:", value=0)
#@ String (visibility=MESSAGE, value="<html>Enter size range (labels within this range are retained):", required=false) msg
#@ Integer min_size(label="Minimum size (pixels):", min=0, value=0)
#@ Integer max_size(label="Maximum size (pixels):", value=0)
#@output ImgPlus output

from net.imglib2.img.display.imagej import ImageJFunctions
from net.imglib2.view import Views
from net.imglib2.util import Util
from net.imglib2.img import Img
from net.imglib2.algorithm.labeling.ConnectedComponents import StructuringElement
from net.imglib2.roi import Regions
from net.imglib2.roi.labeling import LabelRegions, ImgLabeling
from net.imagej.axis import Axes, DefaultLinearAxis
from org.scijava.table import DefaultGenericTable
from de.csbdresden.stardist import StarDist2D


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


def filter_labeling(labeling, min_size, max_size):
    """
    Filter an index image's labels by minimum/maximum pixel size exclusion.

    :param labeling: A labeling created from the index image.
    :param min_size: Minimum label size.
    :param max_size: Maximum label size.
    """
    # get the label regions from the labeling
    index_img = labeling.getIndexImg()
    regs = LabelRegions(labeling)
    for r in regs:
        # get a sample region and compute the size
        sample = Regions.sample(r, index_img)
        size = float(str(ops.stats().size(sample)))
        # if region size is outside of min/max range set to zero
        if size <= float(min_size) or size >= float(max_size):
            remove_label(sample)

    return cs.convert(index_img, ImgLabeling)


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


def segment_nuclei(image):
    """
    Run StarDist2D and return a label.
    """
    # set 3rd axis to time
    image = cs.convert(image, Img)
    t_axis = Axes.get("Time")
    image.setAxis(DefaultLinearAxis(t_axis), 2)

    # run StarDist2D on the timeseries
    res = cmd.run(StarDist2D,
                  False,
                  "input", image,
                  "modelChoice", "Versatile (fluorescent nuclei)",
                  ).get()

    return res.getOutput("label")


def compute_stats(labeling, image):
    # extract regions
    regs = LabelRegions(labeling)
    reg_labels = regs.getExistingLabels()

    # set up table
    table = DefaultGenericTable(3, 0)
    table.setColumnHeader(0, "label")
    table.setColumnHeader(1, "size")
    table.setColumnHeader(2, "MFI")

    i = 0
    for r in regs:
        sample = Regions.sample(r, image)
        table.appendRow()
        table.set("label", i, int(r.getLabel()))
        table.set("size", i, ops.stats().size(sample).getRealDouble())
        table.set("MFI", i, ops.stats().mean(sample).getRealDouble())
        i += 1
    
    return table

# split channels
chs = split_channels(img)

# segment objects
puncta_labeling = segment_puncta(chs[1])
puncta_labeling = filter_labeling(puncta_labeling, min_size, max_size)
nuclei_labeling = segment_nuclei(chs[3])

# compute stats
puncta_seg_results_table = compute_stats(puncta_labeling, chs[1])

# display results
ui.show(puncta_labeling.getIndexImg())
ui.show(puncta_seg_results_table)
ui.show(nuclei_labeling)