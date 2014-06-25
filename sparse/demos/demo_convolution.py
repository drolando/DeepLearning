"""This demo will show how we do simple convolution on the lena image with
a 15*15 average filter.
"""

from decaf import base
from decaf.util import smalldata
from decaf.layers import convolution, fillers
import numpy as np
from skimage import io

"""The main demo code."""
img = np.asarray(smalldata.lena())
img = img.reshape((1,) + img.shape).astype(np.float64)
# wrap the img in a blob
input_blob = base.Blob()
input_blob.mirror(img) #initialize the Blob with the image's data

num_layers = 6

#for i in range(0, num_layers):

# create a convolutional layer
layer = convolution.ConvolutionLayer(
    name='convolution',
    num_kernels=1,
    ksize=15,
    stride=1,
    mode='same',
    filler=fillers.ConstantFiller(value=1./15/15/3))
print layer._kernels._filler.spec

# run the layer
output_blob = base.Blob()
layer.forward([input_blob], [output_blob])

print layer._kernels._filler.spec

input_blob = output_blob


out = output_blob.data()[0, :, :, 0].astype(np.uint8)
io.imsave('out%d_1.png'%num_layers, out)
print('Convolution result written to out%d_1ar.png'%num_layers)
