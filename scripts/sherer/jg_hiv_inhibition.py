#@ OpEnvironment ops
#@ OpService ijops
#@ IOService io
#@ ConvertService cs
#@ ImgPlus img
#@ UIService ui
#@ String (visibility = MESSAGE, value = "<b>Channel settings</b>", required = false) ch_msg
#@ String (label = "Cytoplasmic channel A name", value = "a") ch_a_cyto_name
#@ String (label = "Cytoplasmic channel B name", value = "b") ch_b_cyto_name
#@ String (label = "ZBTB2 Channel C name", value = "c") ch_c_name
#@ String (label = "Nuclear channel name", value = "d") ch_d_nuc_name
#@ Integer (label = "Channel A position", value = 1) ch_a_pos
#@ Integer (label = "Channel B position", value = 2) ch_b_pos
#@ Integer (label = "Channel C position", value = 3) ch_c_pos
#@ Integer (label = "Channel D position", value = 4) ch_d_pos
#@ String (visibility = MESSAGE, value = "<b>Texture settings</b>", required = false) tx_msg
#@ Integer (label = "GLCM Gray Levels", min = 1, value = 128) glcm_gv
#@ Integer (label = "GLCM Distance", min = 1, value = 10) glcm_dist
#@ String (visibility = MESSAGE, value = "<b>Cellpose settings</b>", required = false) cp_msg
#@ String (label="Cellpose Python path:") cp_py_path 
#@ String (label="Pretrained model:", choices={"cyto", "cyto2", "cyto2_cp3", "cyto3", "nuclei", "livecell_cp3", "deepbacs_cp3", "tissuenet_cp3", "bact_fluor_cp3", "bact_phase_cp3", "neurips_cellpose_default", "neurips_cellpose_transformer", "neurips_grayscale_cyto2", "transformer_cp3", "yeast_BF_cp3", "yeast_PhC_cp3"}, style="listBox") model
#@ Float (label="Cytoplasmic diameter:", style="format:0.00", value=69.9, stepSize=0.01) cyto_diameter
#@ Float (label="Nuclear diameter:", style="format:0.00", value=15.0, stepSize=0.01) nuc_diameter
#@ Float (label="Flow threshold:", style="format:0.0", value=0.4, stepSize=0.1) flow_thres
#@ Float (label="Cellprob threshold:", style="format:0.0", value=0.0, stepSize=0.1) cellprob_thres

from net.imglib2.roi import Regions
from net.imglib2.roi.labeling import ImgLabeling, LabelRegions
from net.imglib2.type.numeric.integer import UnsignedShortType

from net.imagej.ops.image.cooccurrenceMatrix import MatrixOrientation2D
from org.scijava.table import DefaultGenericTable

from java.nio.file import Files

import os
import subprocess
import shutil

def extract_channel(image, ch):
    """Extract a channel from the input image.

    Extract the given channel from the input image.

    :param image:

        Input Img.

    :param ch:

        Channel number to extract.

    :return:

        A view of the extracted channel.
    """
    return ops.op("transform.hyperSliceView").input(image, 2, ch - 1).apply()


def glcm(sample):
    """
    """
    orientations = [
        MatrixOrientation2D.ANTIDIAGONAL,
        MatrixOrientation2D.DIAGONAL,
        MatrixOrientation2D.HORIZONTAL,
        MatrixOrientation2D.VERTICAL
        ]
    corr_results = []
    diff_results = []
    for angle in orientations:
        corr = ijops.haralick().correlation(sample, glcm_gv, glcm_dist, angle).getRealDouble()
        diff = ijops.haralick().differenceVariance(sample, glcm_gv, glcm_dist, angle).getRealDouble()
        corr_results.append(corr)
        diff_results.append(diff)
    mean_corr = sum(corr_results) / float(len(corr_results))
    mean_diff = sum(diff_results) / float(len(diff_results))

    return (mean_corr, mean_diff)


def run_cellpose(image, diameter):
    # create a temp directory
    tmp_dir = Files.createTempDirectory("fiji_tmp")

    # save input image
    tmp_save_path = tmp_dir.toString() + "/cellpose_input.tif"
    io.save(image, tmp_save_path)
    
    # create 2D cellpose command
    cp_cmd = "cellpose --dir {} --pretrained_model {} --flow_threshold {} --cellprob_threshold {} --diameter {} --save_tif --no_npy".format(tmp_dir, model, flow_thres, cellprob_thres, diameter)

    # run cellpose
    full_cmd = "{} -m {}".format(cp_py_path, cp_cmd)
    process = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    process.wait()

    # harvest cellpose results
    cp_masks_path = os.path.join(tmp_dir.toString(), "cellpose_input_cp_masks.tif")
    cp_masks = io.open(cp_masks_path)

    # remove temp directory
    shutil.rmtree(tmp_dir.toString())

    return cp_masks


def link_nuclei_cytoplasm_labels(cyto_labeling, nuc_labeling):
    """
    """
    cyto_idx_img = cyto_labeling.getIndexImg()
    nuc_regs = LabelRegions(nuc_labeling)
    linked_samples = {}
    
    for r in nuc_regs:
        s = Regions.sample(r, cyto_idx_img)
        v = ops.op("stats.median").input(s).apply().getRealDouble()
        linked_samples[r.getLabel()] = int(v)

    return linked_samples


def run(image):
    """Run the pipeline.
    """
    # a = a3g
    # b = mche
    # c = cfp
    # d = nuc

    # extract channels
    ch_a = extract_channel(image, ch_a_pos)
    ch_b = extract_channel(image, ch_b_pos)
    ch_c = extract_channel(image, ch_c_pos)
    ch_d = extract_channel(image, ch_d_pos)
    
    # add cytoplasmic channels
    ch_ab = ops.op("create.img").input(ch_a, UnsignedShortType()).apply() 
    ops.op("math.add").input(ch_a, ch_b).output(ch_ab).compute()

    # create cellpose labelings
    print("[INFO]: Running Cellpose on cytoplasmic channel...")
    labelings_ab = cs.convert(run_cellpose(ch_ab, cyto_diameter), ImgLabeling)
    print("[INFO]: Running Cellpose on nuclear channel...")
    labelings_d = cs.convert(run_cellpose(ch_d, nuc_diameter), ImgLabeling)

    linked = link_nuclei_cytoplasm_labels(labelings_ab, labelings_d)
    
    # create result table
    table = DefaultGenericTable(10, 0)
    table.setColumnHeader(0, "nucleus #")
    table.setColumnHeader(1, "{} nuc MFI".format(ch_b_cyto_name))
    table.setColumnHeader(2, "{} cyto MFI".format(ch_b_cyto_name))
    table.setColumnHeader(3, "N/C ratio {}".format(ch_b_cyto_name))
    table.setColumnHeader(4, "{} nuc MFI".format(ch_c_name))
    table.setColumnHeader(5, "{} cyto MFI".format(ch_c_name))
    table.setColumnHeader(6, "N/C ratio {}".format(ch_c_name))
    table.setColumnHeader(7, "{} total MFI".format(ch_a_cyto_name))
    table.setColumnHeader(8, "{} GLCM correlation".format(ch_b_cyto_name))
    table.setColumnHeader(9, "{} GLCM difference variance".format(ch_b_cyto_name))
    regs_ab = LabelRegions(labelings_ab) # cyto
    regs_d = LabelRegions(labelings_d) # nuc
    i = 0
    for k in linked.keys():
        nuc_reg = regs_d.getLabelRegion(int(k))
        cyto_id = linked[k]
        cyto_reg = regs_ab.getLabelRegion(cyto_id)
        s_cyto_a = Regions.sample(cyto_reg, ch_a) # yfp
        s_cyto_b = Regions.sample(cyto_reg, ch_b)
        s_cyto_c = Regions.sample(cyto_reg, ch_c)
        s_nuc_b = Regions.sample(nuc_reg, ch_b)
        s_nuc_c = Regions.sample(nuc_reg, ch_c)

        # measurements
        ch_a_cyto_mfi = ijops.stats().mean(s_cyto_a).getRealDouble()
        ch_b_nuc_mfi = ijops.stats().mean(s_nuc_b).getRealDouble()
        ch_b_cyto_mfi = ijops.stats().mean(s_cyto_b).getRealDouble()
        ch_c_nuc_mfi = ijops.stats().mean(s_nuc_c).getRealDouble()
        ch_c_cyto_mfi = ijops.stats().mean(s_cyto_c).getRealDouble()
        ch_b_glcm = glcm(s_cyto_b) # returns a tuple
        
        # populate the table
        table.appendRow()
        table.set("nucleus #", i, k) # nucleus id
        table.set("{} nuc MFI".format(ch_b_cyto_name), i, ch_b_nuc_mfi)
        table.set("{} cyto MFI".format(ch_b_cyto_name), i, ch_b_cyto_mfi)
        table.set("N/C ratio {}".format(ch_b_cyto_name), i, ch_b_nuc_mfi / ch_b_cyto_mfi)
        table.set("{} nuc MFI".format(ch_c_name), i, ch_c_nuc_mfi)
        table.set("{} cyto MFI".format(ch_c_name), i, ch_c_cyto_mfi)
        table.set("N/C ratio {}".format(ch_c_name), i, ch_c_nuc_mfi / ch_c_cyto_mfi)
        table.set("{} total MFI".format(ch_a_cyto_name), i, ch_a_cyto_mfi)
        table.set("{} GLCM correlation".format(ch_b_cyto_name), i, ch_b_glcm[0])
        table.set("{} GLCM difference variance".format(ch_b_cyto_name), i, ch_b_glcm[1])
        i += 1
    
    ui.show(table)
    ui.show("cyto", labelings_ab.getIndexImg()) 
    ui.show("nuc", labelings_d.getIndexImg())

run(img)
print("[INFO]: Done!")
