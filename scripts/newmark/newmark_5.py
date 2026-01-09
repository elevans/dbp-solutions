#@ ImageJ ij
#@ OpEnvironment ops
#@ ConvertService cs
#@ IOService io
#@ UIService ui
#@ Img (label = "Input image:", autofill = false) img
#@ String (visibility = MESSAGE, value ="<b>[ Channel selection settings ]</b>", required = false) ch_msg
#@ String (label = "Puncta channel number:", choices = {"None", "1", "2", "3", "4", "5"}, style = "listBox", value = "None") p_ch
#@ String (label = "Nuclear channel number:", choices = {"None", "1", "2", "3", "4", "5"}, style = "listBox", value = "None") n_ch
#@ String (label = "Marker channel number:", choices = {"None", "1", "2", "3", "4", "5"}, style = "listBox", value = "None") m_ch
#@ String (visibility = MESSAGE, value ="<b>[ Puncta segmentation settings ]</b>", required = false) pun_msg 
#@ Double (label = "High pass filter Sigma:", style = "format:0.00", min = 0.0, value = 5.0) hp_sigma
#@ Double (label = "Watershed Sigma:", style = "format:0.00", min = 0.0, value = 2.0) ws_sigma
#@ String (label = "Global threshold method:", choices={"huang", "ij1", "intermodes", "isoData", "li", "maxEntropy", "maxLikelihood", "mean", "minError", "minimum", "moments", "otsu", "percentile", "renyiEntropy", "rosin", "shanbhag", "triangle", "yen"}, style="listBox") threshold
#@ Integer (label = "Minimum puncta size (pixels):", min = 0, value = 0) min_size
#@ Integer (label = "Maximum puncta size (pixels):", min = 0, value = 0) max_size
#@ String (visibility = MESSAGE, value ="<b>[ Cellpose settings ]</b>", required = false) cp_msg
#@ Float (label = "Diameter:", style = "format:0.00", min = 0.0, value = 30.0) diameter
#@ Float (label = "Flow threshold:", style = "format:0.00", min = 0.0, value = 4.0) flow_threshold
#@ Float (label = "Cell probability threshold:", style = "format:0.00", min = 0.0, value = 0.0) cellprob_threshold
#@ String (visibility = MESSAGE, value ="<b>[ Results settings ]</b>", required = false) rslt_msg
#@ Boolean (label = "Show puncta label image:", value = false) pun_show
#@ Boolean (label = "Show nuclear label image:", value = false) nuc_show
#@ Boolean (label = "Show marker label image:", value = fale) mar_show
#@ Boolean (label = "Show puncta results table:", value = true) show_puncta_results
#@ Boolean (label = "Show nuclei results table:", value = true) show_nuclei_results
#@output Img output

from collections import Counter
import scyjava as sj
from cellpose import models, core, io, plot

# get java resources
DefaultGenericTable = sj.jimport("org.scijava.table.DefaultGenericTable")
ImgLabeling = sj.jimport("net.imglib2.roi.labeling.ImgLabeling")
LabelRegions = sj.jimport("net.imglib2.roi.labeling.LabelRegions")
PointRoi = sj.jimport("ij.gui.PointRoi")
Regions = sj.jimport("net.imglib2.roi.Regions")

def cellpose(image):
    arr = ij.py.to_xarray(image).data
    model = models.CellposeModel(gpu=True)
    mask, flow, styles = model.eval(arr,
    			z_axis=0,
    			diameter=diameter,
    			flow_threshold=flow_threshold,
    			cellprob_threshold=cellprob_threshold,
    			do_3D=True)

    return mask


def center_to_roi_manager(labeling):
    """Add the center of mass to the ROI manager.

    :param labeling: The input ImgLabeling.
    """
    idx_img = labeling.getIndexImg()
    regs = LabelRegions(labeling)
    rm = ij.RoiManager().getRoiManager()
    for r in regs:
        # get label ID
        r_id = Regions.sample(r, idx_img).firstElement()
        pos = r.getCenterOfMass().positionAsDoubleArray()
        roi = PointRoi(pos[0], pos[1])
        # set the Z position of the ROI
        roi.setName(str(r_id))
        roi.setPosition(0, int(round(pos[2])), 1)
        rm.addRoi(roi)


def extract_channel(image, channel, axis = 2):
    """Extract a channel from the input image
        
    :param image: The input image to slice.
    :param channel: The channel to extract.
    :prama axis: The axis of the channel dimension.
    :return: A slice of the image with the given channel.
    """
    v = int(str(channel))
    return ij.op().transform().hyperSliceView(image, axis, v - 1)


def find_most_common_value(sample):
    vals = []
    c = sample.cursor()
    while c.hasNext():
        c.fwd()
        v = c.get().getInteger()
        if v > 0:
            vals.append(v)

    # count the most common element in vals array
    if vals:
        return Counter(vals).most_common(1)[0][0]
    else:
        return "None"


def gaussian_high_pass_filter(image, sigma):
    """Apply a Gaussian high pass filter.

    :param image: The input image.
    :param sigma: The sigma value.
    :return: High pass filtered image.
    """
    img_f = ij.op().convert().float32(image)
    gauss_image = ij.op().filter().gauss(img_f, sigma)

    return ij.op().math().subtract(img_f, gauss_image)


def measurements(channels, puncta_labeling, nuclei_labeling):
    # initialize the results tables
    p_table = DefaultGenericTable(4, 0)
    n_table = DefaultGenericTable(4, 0)


    # set up puncta table headers
    p_table.setColumnHeader(0, "foci ID")
    p_table.setColumnHeader(1, "cell ID")
    p_table.setColumnHeader(2, "foci MFI")
    p_table.setColumnHeader(3, "foci size (pixels)")

    # set up nuclei table headers
    n_table.setColumnHeader(0, "cell ID")
    n_table.setColumnHeader(1, "marker MFI")
    n_table.setColumnHeader(2, "marker size (pixels)")
    n_table.setColumnHeader(3, "foci count")

    # extract puncta and nuclei regions
    p_regions = LabelRegions(puncta_labeling)
    n_regions = LabelRegions(nuclei_labeling)

    # extract puncta and nuclei index images (for linking)
    p_idx_img = puncta_labeling.getIndexImg()
    n_idx_img = nuclei_labeling.getIndexImg()

    # nuclei/puncta map
    nuc_pun_count = {}
    # create puncta results table
    i = 0
    for p in p_regions:
        # sample: puncta of puncta index image -> puncta ID
        p_pi_sample = Regions.sample(p, p_idx_img)
        # sample: puncta of nuclei index image -> nuclei ID
        p_ni_sample = Regions.sample(p, n_idx_img)
        # sample: punta of puncta raw -> intensity measurements
        p_pr_sample = Regions.sample(p, channels.get("pun"))
        p_id = p_pi_sample.firstElement()
        # be smarter about nuclei detection
        n_id = find_most_common_value(p_ni_sample)
        if n_id not in nuc_pun_count.keys():
            nuc_pun_count[n_id] = 1
        else:
            nuc_pun_count[n_id] += 1
        # measure puncta intensity
        p_mfi = ij.op().stats().mean(p_pr_sample).getRealDouble()
        p_size = ij.op().stats().size(p_pr_sample).getRealDouble()
        # construct puncta table
        p_table.appendRow()
        p_table.set("foci ID", i, p_id)
        p_table.set("cell ID", i, n_id)
        p_table.set("foci MFI", i, p_mfi)
        p_table.set("foci size (pixels)", i, p_size)
        i += 1

    # create nuclei results table
    i = 0
    for n in n_regions:
        # sample: nuclei of nuclei index -> nuclei ID
        n_ni_sample = Regions.sample(n, n_idx_img)
        # sample: nuclei of nuclei raw -> intensity measurements
        n_nr_sample = Regions.sample(n, channels.get("nuc"))
        n_id = n_ni_sample.firstElement().getInteger()
        n_mfi = ij.op().stats().mean(n_nr_sample).getRealDouble()
        n_size = ij.op().stats().size(n_nr_sample).getRealDouble()
        # construct nuclei table
        n_table.appendRow()
        n_table.set("cell ID", i, n_id)
        n_table.set("marker MFI", i, n_mfi)
        n_table.set("marker size (pixels)", i, n_size)
        n_table.set("foci count", i, nuc_pun_count.get(n_id))
        i += 1

    return (p_table, n_table)


def size_filter_labeling(labeling, min_size, max_size=None):
    """Apply a size filter to an ImgLabeling.

    :param labeling: Input ImgLabeling.
    :param min_size: Minimum number of label pixels. If None, then the minimum
        value is set to 0.
    :param max_size: Maximum number of label pixels. If None, then the maximum
        value is set to dim lengths X * Y.
    """
    # get label image from the labeling
    idx_img = labeling.getIndexImg()

    # set max/min defaults if necessary
    if not min_size:
        min_size = 0
    if not max_size:
        dims = idx_img.shape
        max_size = dims[0] * dims[1]

    # get a sample over the label image and remove out-of-bounds
    regs = LabelRegions(labeling)
    for r in regs:
        s = Regions.sample(r, idx_img)
        l = int(ij.op().stats().size(s).getRealDouble())
        if l <= min_size or l >= max_size:
            remove_label(s)

    return ij.convert().convert(idx_img, ImgLabeling)


def remove_label(sample):
    c = sample.cursor()
    while c.hasNext():
        c.fwd()
        c.get().set(0)


def run(image):
    """Pipeline entry point.

    :param image: The input image.
    """
    # extract specified channels
    print("[INFO]: Extracting images...")
    chs = {}
    if p_ch != "None":
        chs["pun"] = (extract_channel(image, p_ch))
    if n_ch != "None":
        chs["nuc"] = (extract_channel(image, n_ch))
    if m_ch != "None":
        chs["mar"] = (extract_channel(image, m_ch))

    # segment puncta channel and add to ROI manager
    if p_ch != "None":
    	print("[INFO]: Processing the foci channel...")
    	pun_img_hp = gaussian_high_pass_filter(chs.get("pun"), hp_sigma)
    	pun_img_ths = ij.op().run(f"threshold.{threshold}", pun_img_hp)
    	pun_img_ws = ij.op().image().watershed(None, pun_img_ths, False, False, ws_sigma, pun_img_ths)
    	pun_img_ws = size_filter_labeling(pun_img_ws, min_size, max_size)
    	center_to_roi_manager(pun_img_ws)
    	if pun_show:
    		ui.show("puncta label image", pun_img_ws.getIndexImg())

    # segment nuclear channel
    if n_ch != "None":
    	print("[INFO]: Running cellpose on nuclear channel...")
    	nuc_img_gb = ij.op().filter().gauss(chs.get("nuc"), 2.0)
    	nuc_msk = cellpose(nuc_img_gb)
    	if nuc_show:
    		ui.show(ij.py.to_dataset(nuc_msk))

    # segment marker channel
    if m_ch != "None":
    	print("[INFO]: Running cellpose on marker channel...")
    	mar_mask = cellpose(chs.get("mar"))
    	if mar_show:
    	    ui.show(ij.py.to_dataset(mar_mask))

    # run measurements
    print("[INFO]: Calculating measurements...")
    results = measurements(chs, pun_img_ws, ij.convert().convert(ij.py.to_dataset(nuc_msk), ImgLabeling))
    if show_puncta_results:
    	ui.show("foci results table", results[0])
    if show_nuclei_results:
    	ui.show("nuclei results table", results[1])

run(img)
print("[INFO]: Done!")
