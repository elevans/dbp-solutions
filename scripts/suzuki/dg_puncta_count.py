#@ OpEnvironment ops
#@ ConvertService cs
#@ UIService ui
#@ Img (label = "Input image:", autofill = false) img
#@ Float (label = "Gaussian blur sigma:", style = "format:0.00", min = 0.0, value = 0.0) sigma
#@ String (label="Global threshold method:", choices={"huang", "ij1", "intermodes", "isoData", "li", "maxEntropy", "maxLikelihood", "mean", "minError", "minimum", "moments", "otsu", "percentile", "renyiEntropy", "rosin", "shanbhag", "triangle", "yen"}, style="listBox", value = "triangle") thresh_method

from net.imglib2.algorithm.labeling.ConnectedComponents import StructuringElement
from net.imglib2.roi.labeling import ImgLabeling, LabelRegions
from net.imglib2.type.numeric.real import FloatType
from net.imglib2.type.logic import BitType

from org.scijava.table import DefaultGenericTable

# apply gaussian blur high pass filter
print("[INFO]: Applying gaussian blur {}...".format(sigma))
gauss_img = ops.op("create.img").input(img, FloatType()).apply()
ops.op("filter.gauss").input(img, sigma).output(gauss_img).compute()
print("[INFO]: Subtracting images...")
sub_img = ops.op("create.img").input(img, FloatType()).apply()
ops.op("math.sub").input(img, gauss_img).output(sub_img).compute()

# apply threshold
print("[INFO]: Appying {} theshold...".format(thresh_method))
thresh_img = ops.op("create.img").input(img, BitType()).apply()
ops.op("threshold.{}".format(thresh_method)).input(sub_img).output(thresh_img).compute()

# run CCA (four connected)
labeling = ops.op("labeling.cca").input(thresh_img, StructuringElement.FOUR_CONNECTED).apply()

# create table and populate data
table = DefaultGenericTable(2, 0)
table.setColumnHeader(0, "ID")
table.setColumnHeader(1, "size (pixel)")
regs = LabelRegions(labeling)
i = 0
for r in regs:
    table.appendRow()
    table.set("ID", i, r.getLabel())
    table.set("size (pixel)", i, ops.op("stats.size").input(r).apply())
    i += 1

# show outputs
ui.show("Results table", table)
ui.show("Label image", labeling.getIndexImg())
