import os
import imagej
import numpy as np
import scyjava as sj
from skimage.filters import threshold_otsu as otsu
from statistics import mean
from typing import List, Sequence

# initialize ImageJ
ij = imagej.init(mode='interactive')
print(f"ImageJ2 version: {ij.getVersion()}")


def get_file_paths(dir: str, ext: str) -> List[str]:
    """Get file paths.
    """
    paths = []
    abs_dir = os.path.abspath(dir)
    for root, dirs, file_names in os.walk(abs_dir):
        file_names.sort()
        for name in file_names:
            # check for file extension
            if not name.endswith(ext):
                continue
            paths.append(os.path.join(root, name))

    return paths


def load_path_images(paths: Sequence[str], ch: int) -> List:
    """Load images
    """
    images = []
    for p in paths:
        print(f"[STATUS]: Loading image: {os.path.basename(p)}")
        images.append(ij.io().open(p)[:, :, ch])

    return images


def mean_histograms(image_list: List):
    """
    Compute the mean histogram
    """
    histograms = []
    bins = 65536
    # compute individual histograms
    print(f"[STATUS]: Computing individual histograms...")
    for i in image_list:
        histograms.append(ij.op().image().histogram(i, bins).toLongArray())
    
    # compute mean histogram
    mean_histogram = []
    print(f"[STATUS]: Computing mean histogram...")
    for b in range(bins):
        temp = []
        for h in histograms:
            temp.append(h[b])
        mean_histogram.append((mean(temp)))
    res_histogram, res_bin_edges = np.histogram(mean_histogram, bins=bins)

    return res_histogram


def create_output_mask_stack(image_list: List, threshold: int, paths: str, output_dir:str, ch: int):
    """Create output mask stack and save.
    """
    # convert slices to numpy array
    narrs = []
    for img in image_list:
        narrs.append(ij.py.from_java(img))
    # get masks for each numpy array
    masks = []
    for narr in narrs:
        masks.append(narr > threshold)
    # stack the source and mask arrays
    res = []
    print(f"[STATUS]: Creating image stacks...")
    for i in range(len(narrs)):
        mask = masks[i].astype(np.uint16) * 65535 # convert bool mask to uint16
        xarr = ij.py.to_xarray(np.dstack([narrs[i], mask]), dim_order=['row', 'col', 'ch'])
        res.append(ij.py.to_dataset(xarr))
    # save output
    print(f"[STATUS]: Saving output image stacks...")
    for p, ds in zip(paths, res):
        fname = os.path.basename(p)
        out_path = os.path.join(output_dir, fname.replace(".tif", f"_ch{ch + 1}_output.tif"))
        ij.io().save(ds, out_path)


def main(input_dir, output_dir, ch, bins=65536):
    paths = get_file_paths(input_dir, ".tif")
    images = load_path_images(paths, ch)
    mean_histogram = mean_histograms(images)
    thres = otsu(hist=mean_histogram, nbins=bins)
    print(f"\n[Otsu Threshold]: {thres}\n[Channel]: {ch}\n[Bins]: {bins}\n")
    create_output_mask_stack(images, thres, paths, output_dir)


if __name__ == "__main__":
    main(input("Data directory: "), input("Output directory: "), int(input("Channel: ")))