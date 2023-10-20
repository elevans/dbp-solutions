#@ DatasetIOService ds
#@ ConvertService cs
#@ UIService ui
#@ OpService ops
#@ File (label = "Input directory", style="directory") in_dir
#@ String (label = "File extension", value=".tif") ext
#@ String (choices={"otsu", "huang", "moments", "renyiEntropy", "triangle", "yen"}, style="listBox") thres_method
#@ Integer (label = "Channel", value=1) ch

import os
from net.imglib2.histogram import Histogram1d
from net.imglib2.histogram import Integer1dBinMapper
from net.imglib2.type.numeric.integer import IntType
from ij.gui import GenericDialog

def _mean(int_list):
    total_sum = sum(int_list)
    count = len(int_list)
    return IntType(total_sum / count)

def run():
    # extract image paths to a list
    paths = []
    data_dir = in_dir.getAbsolutePath()
    for root, dirs, file_names in os.walk(data_dir):
        file_names.sort()
        for name in file_names:
            # check for file extension
            if not name.endswith(ext):
                continue
            paths.append(os.path.join(root, name))

    # open each image slice and save as list
    images = []
    for p in paths:
        dataset = ds.open(p)
        dims = dataset.dimensionsAsLongArray()
        view = ops.transform().intervalView(dataset, [0, 0, ch - 1], [dims[0] - 1, dims[1] - 1, ch - 1])
        images.append(view)

    # compute histograms
    histograms = []
    for i in images:
        histograms.append(ops.image().histogram(i).toLongArray())

    # compute average histogram
    mean_histogram = []
    bins = 256
    for b in range(bins):
        temp = []
        for h in histograms:
            temp.append(h[b])
        mean_histogram.append((_mean(temp)))

    bin_mapper = Integer1dBinMapper((min(mean_histogram)).getIntegerLong(), 256, False)
    res_histogram = Histogram1d(mean_histogram, bin_mapper)
    res_histogram.addData(mean_histogram)

    # apply the selected threshold to the mean histogram
    #thres_val = ops.run("threshold.{}".format(thres_method), res_histogram)
    
    return histograms

res = run()
#gd = GenericDialog("Mean histogram threshold output")
#gd.addNumericField("Threshold {} value:".format(thres_method), res.int, 0)
#gd.showDialog()