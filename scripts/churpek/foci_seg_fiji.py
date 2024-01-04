#@ ImgPlus img
#@ OpService ops
#@ ConvertService cs
#@ CommandService cmd
#@ UIService ui
#@ Boolean (label="Show ROIs:", value=False) show

from ij.plugin.frame import RoiManager
from ij.gui import PolygonRoi
from net.imglib2.algorithm.labeling.ConnectedComponents import StructuringElement
from net.imglib2.algorithm.math import ImgMath
from net.imglib2.roi.labeling import LabelRegions, ImgLabeling
from net.imglib2.roi import Regions
from net.imglib2.algorithm.neighborhood import HyperSphereShape
from org.scijava.table import DefaultGenericTable

def split_channels(image, ch_axis=2):
    channels = []
    for i in range(image.dimensionsAsLongArray()[ch_axis]):
        channels.append(ops.transform().hyperSliceView(image, ch_axis, i))
    
    return channels


def label_img_to_roi_manager(labeling):
    contours = []
    regions = LabelRegions(labeling)
    # apply contour op to each region
    for r in regions:
        contours.append(ops.geom().contour(r, True))

    # add labels to the roi manager
    rm = RoiManager.getRoiManager()
    for c in contours:
        rm.addRoi(cs.convert(c, PolygonRoi))

    
def segment_nuclei(image):
    # segment params
    shape = HyperSphereShape(2)
    use_eight_connectivity = True
    draw_watersheds = False
    sigma = 30

    # threshold and cleanup nuclei
    nuc_seg = ops.threshold().huang(image)
    nuc_seg = ops.morphology().open(nuc_seg, [shape])
    nuc_seg = ops.morphology().fillHoles(nuc_seg)
    nuc_label = ops.create().imgLabeling(nuc_seg)

    # waterhsed and create ImgLabeling
    ops.image().watershed(nuc_label, nuc_seg,use_eight_connectivity,draw_watersheds,sigma,nuc_seg)
    
    return nuc_label


def segment_puncta(image):
    # segment params
    sigma = 13

    image_f = ops.convert().float32(image)
    image_g = ops.filter().gauss(image_f, sigma)
    # TODO: replace copy with an empty image
    puncta_seg = ImgMath.computeInto(ImgMath.sub(image_f, image_g), image_f.copy())
    puncta_seg = ops.threshold().yen(puncta_seg)
    puncta_label = ops.labeling().cca(puncta_seg, StructuringElement.EIGHT_CONNECTED)

    return puncta_label


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


def get_sample_labels(sample):
    # get the sample region's cursor
    c = sample.cursor()
    vals = []
    # get all unique values in the sample
    while c.hasNext():
        c.fwd()
        vals.append(int(c.get().toString()))
    
    # remove the background label
    vals = list(set(vals))
    if 0 in vals:
        vals.remove(0)
   
    return vals


def run(nuc_labeling, tar_labeling, filter=True):
    """Run the analysis component of the script.

    :param nuc_labeling: Input nucleus ImgLabeling.
    :param tar_labeling: Input target ImgLabeling.
    :param filter: Filter out ROIs that our outside the nucleus.
    """
    nuc_index_img = nuc_labeling.getIndexImg()
    tar_index_img = tar_labeling.getIndexImg()
    nuc_regions = LabelRegions(nuc_labeling)
    tar_regions = LabelRegions(tar_labeling)
    nuc_linked_puncta = [] # store nucleus linked for clearing

    #set up table
    table = DefaultGenericTable(3, 0)
    table.setColumnHeader(0, "nucleus_label")
    table.setColumnHeader(1, "puncta_label")
    table.setColumnHeader(2, "area")

    i = 0
    for nr in nuc_regions:
        # get samples
        tar_sample = Regions.sample(nr, tar_index_img)
        # get unique labels within nuclear region
        tar_labels = get_sample_labels(tar_sample)
        for j in range(len(tar_labels)):
            l = tar_labels[j]
            nuc_linked_puncta.append(l)
            table.appendRow()
            table.set("nucleus_label", i, int(nr.getLabel()))
            table.set("puncta_label", i, l)
            p_sample = Regions.sample(tar_regions.getLabelRegion(l), tar_index_img)
            area = float(ops.stats().size(p_sample).getRealDouble())
            table.set("area", i, area)
            i += 1
            continue
    
    # set labels outside the nucleus to 0 in the index image
    if filter:
        for tr in tar_regions:
            tr_sample = Regions.sample(tr, tar_index_img)
            l = get_sample_labels(tr_sample)[0]
            if l not in nuc_linked_puncta:
                remove_label(tr_sample)

        # overwrite tar_labeling with new ImgLabeling
        tar_labeling = cs.convert(tar_index_img, ImgLabeling)

    return table

# spilt input image into channels
chs = split_channels(img)

# segment the image
nuc_labels = segment_nuclei(chs[2])
puncta_labels = segment_puncta(chs[0])

# run
results = run(nuc_labels, puncta_labels)

if show:
    label_img_to_roi_manager(nuc_labels)
    label_img_to_roi_manager(puncta_labels)
    ui.show(results)