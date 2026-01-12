#@ OpService ops
#@ UIService ui
#@ Img (label = "Input image:", autofill = false) img
#@ Integer (label = "Mean filter radius:", min = 0, value = 6) radius
#@ String (choices={"Diamond", "Hyper Sphere"}, style="listBox", value = "Hyper Sphere") shape
#@ Float (label="Gaussian blur Sigma:", style="format:0.00", min=0.0, value=7.00) sigma
#@output Img result

from net.imglib2.algorithm.neighborhood import (DiamondShape,
                                                HyperSphereShape)

shape_map = {
    "Diamond": DiamondShape,
    "Hyper Sphere": HyperSphereShape,
}

def suppress_background(image):
    # apply mean filter
    image = ops.convert().float32(image)
    image_mean = ops.create().img(image)
    ops.filter().mean(image_mean, img, shape_map.get(shape)(radius))
    image = ops.math().multiply(image, image_mean)
    # apply high pass Gaussian filter
    image_gauss = ops.filter().gauss(image, sigma)

    return ops.math().subtract(image, image_gauss)


result = suppress_background(img)
