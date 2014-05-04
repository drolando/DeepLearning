from sparse import base
from sparse.util import smalldata, visualize
from sparse.util import translator, transform
from sparse.layers import core_layers, convolution, fillers, regularization
from sparse.opt import core_solvers
import numpy as np

_JEFFNET_FLIP = True
INPUT_DIM = 227

#images = np.load('/home/daniele/Documents/project/sparse-release/sparse/util/_data/images.npy')
def oversample(image, center_only=False):
    """Oversamples an image. Currently the indices are hard coded to the
        4 corners and the center of the image, as well as their flipped ones,
        a total of 10 images.

        Input:
        image: an image of size (256 x 256 x 3) and has data type uint8.
        center_only: if True, only return the center image.
        Output:
        images: the output of size (10 x 227 x 227 x 3)
    """
    indices = [0, 256 - INPUT_DIM]
    center = int(indices[1] / 2)
    if center_only:
        return np.ascontiguousarray(
            image[np.newaxis, center:center + INPUT_DIM,
            center:center + INPUT_DIM], dtype=np.float32)
    else:
        images = np.empty((10, INPUT_DIM, INPUT_DIM, 3),
            dtype=np.float32)
        curr = 0
        for i in indices:
            for j in indices:
                images[curr] = image[i:i + INPUT_DIM,
                    j:j + INPUT_DIM]
                curr += 1
        images[4] = image[center:center + INPUT_DIM,
        center:center + INPUT_DIM]
        # flipped version
        images[5:] = images[:5, ::-1]
        return images

def classify(image, center_only=False):
    """Classifies an input image.

       Input:
        image: an image of 3 channels and has data type uint8. Only the
            center region will be used for classification.
       Output:
        scores: a numpy vector of size 1000 containing the
            predicted scores for the 1000 classes.
    """
    # first, extract the 256x256 center.
    image = transform.scale_and_extract(transform.as_rgb(image), 256)
    # convert to [0,255] float32
    image = image.astype(np.float32) * 255.
    if _JEFFNET_FLIP:
        # Flip the image if necessary, maintaining the c_contiguous order
        image = image[::-1, :].copy()
    # subtract the mean
    #image -= self._data_mean
    # oversample the images
    images = oversample(image, center_only)
    return images

lena = smalldata.lena()
images = classify(lena)
'''
    now the image is (10, 227, 227, 3)
    10 sub-images
    227 x 227 dimension
    3 rgb
'''


net = base.Net()
'''
    NdarrayDataLayer -- takes a bunch of data and then emits them as Blobs.
    sources - list of images (10, 227, 227, 3)
    forward - puts these images in Blobs and sends them to the next layer
'''
net.add_layer(
    core_layers.NdarrayDataLayer(
        name='input',
        sources=[images]
    ),
    group=1,
    provides='data'
)
'''
    forward - (10, 57, 57, 96)
'''
net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv1',
        num_kernels=96,
        ksize=11,
        stride=4,
        mode='same',
        filler=fillers.GaussianRandFiller()
    ),
    group=1,
    needs='data',
    provides='conv1_cudanet_out'
)
'''
    applies ReLU activation function
    all _data < 0 become zero
    forward - (10, 57, 57, 96)
'''
net.add_layer(
    core_layers.ReLULayer(
        name='conv1_neuron',
        freeze=False
    ),
    group=1,
    needs='conv1_cudanet_out',
    provides='conv1_neuron_cudanet_out'
)
'''
    forward - (10, 28, 28, 96)
'''
net.add_layer(
    core_layers.PoolingLayer(
        name='pool1',
        mode='max',
        psize=3,
        stride=2
    ),
    group=1,
    needs='conv1_neuron_cudanet_out',
    provides='pool1_cudanet_out'
)
net.add_layer(
    core_layers.LocalResponseNormalizeLayer(
        name='rnorm1',
        alpha=0.0001,
        beta=0.75,
        size=5,
        k=1 # in imageNet k=2
    ),
    group=1,
    needs='pool1_cudanet_out',
    provides='rnorm1_cudanet_out'
)
#---------------------------------------------------------
net.add_layer(
    core_layers.SparseFilteringLayer(
        name='sparse_filter_1'
    ),
    group=1,
    needs='rnorm1_cudanet_out',
    provides='sparse1'
)
#---------------------------------------------------------
'''
    2 groups by 128 kernels --> 256
    forward - (10, 28, 28, 256)
'''
net.add_layer(
    core_layers.GroupConvolutionLayer(
        name='conv2',
        stride=1,
        num_kernels=128,
        pad=2,
        ksize=5,
        group=2,
        filler=fillers.GaussianRandFiller()
    ),
    #needs='rnorm1_cudanet_out',
    needs='sparse1',
    provides='conv2_cudanet_out'
)
'''
    forward - (10, 28, 28, 256)
'''
net.add_layer(
    core_layers.ReLULayer(
        name='conv2_neuron',
        freeze=False
    ),
    needs='conv2_cudanet_out',
    provides='conv2_neuron_cudanet_out'
)
'''
    forward - (10, 14, 14, 256)
'''
net.add_layer(
    core_layers.PoolingLayer(
        name='pool2',
        mode='max',
        psize=3,
        stride=2
    ),
    needs='conv2_neuron_cudanet_out',
    provides='pool2_cudanet_out'
)
'''
    forward - (10, 14, 14, 256)
'''
net.add_layer(
    core_layers.LocalResponseNormalizeLayer(
        name='rnorm2',
        alpha=0.0001,
        beta=0.75,
        size=5,
        k=1
    ),
    needs='pool2_cudanet_out',
    provides='rnorm2_cudanet_out'
)
'''
    forward - (10, 14, 14, 384)
'''
net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv3',
        num_kernels=384,
        ksize=3,
        stride=1,
        mode='same',
        filler=fillers.GaussianRandFiller()
    ),
    needs='rnorm2_cudanet_out',
    provides='conv3_cudanet_out'
)
'''
    forward - (10, 14, 14, 384)
'''
net.add_layer(
    core_layers.ReLULayer(
        name='conv3_neuron',
        freeze=False
    ),
    needs='conv3_cudanet_out',
    provides='conv3_neuron_cudanet_out'
)
'''
    2 groups by 192 kernels --> 384
    forward - (10, 14, 14, 384)
'''
net.add_layer(
    core_layers.GroupConvolutionLayer(
        name='conv4',
        stride=1,
        num_kernels=192,
        pad=1,
        ksize=3,
        group=2,
        filler=fillers.GaussianRandFiller()
    ),
    needs='conv3_neuron_cudanet_out',
    provides='conv4_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='conv4_neuron',
        freeze=False
    ),
    needs='conv4_cudanet_out',
    provides='conv4_neuron_cudanet_out'
)
net.add_layer(
    core_layers.GroupConvolutionLayer(
        name='conv5',
        stride=1,
        num_kernels=128,
        pad=1,
        ksize=3,
        group=2
    ),
    needs='conv4_neuron_cudanet_out',
    provides='conv5_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='conv5_neuron',
        freeze=False
    ),
    needs='conv5_cudanet_out',
    provides='conv5_neuron_cudanet_out'
)
net.add_layer(
    core_layers.PoolingLayer(
        name='pool5',
        mode='max',
        psize=3,
        stride=2
    ),
    needs='conv5_neuron_cudanet_out',
    provides='pool5_cudanet_out'
)
net.add_layer(
    core_layers.FlattenLayer(
        name='fc6_flatten',
    ),
    needs='pool5_cudanet_out',
    provides='_sparse_fc6_flatten_out'
)
net.add_layer(
    core_layers.InnerProductLayer(
        name='fc6',
        num_output=4096,
        has_bias=True
    ),
    needs='_sparse_fc6_flatten_out',
    provides='fc6_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='fc6_neuron',
        freeze=False
    ),
    needs='fc6_cudanet_out',
    provides='fc6_neuron_cudanet_out'
)
net.add_layer(
    core_layers.DropoutLayer(
        name='fc6dropout',
        ratio=0.500000,
    ),
    needs='fc6_neuron_cudanet_out',
    provides='fc6dropout_cudanet_out'
)
net.add_layer(
    core_layers.InnerProductLayer(
        name='fc7',
        num_output=4096,
        has_bias=True
    ),
    needs='fc6dropout_cudanet_out',
    provides='fc7_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='fc7_neuron',
        freeze=False
    ),
    needs='fc7_cudanet_out',
    provides='fc7_neuron_cudanet_out'
)
net.add_layer(
    core_layers.DropoutLayer(
        name='fc7dropout',
        ratio=0.500000,
    ),
    needs='fc7_neuron_cudanet_out',
    provides='fc7dropout_cudanet_out'
)
net.add_layer(
    core_layers.InnerProductLayer(
        name='fc8',
        num_output=1000,
        has_bias=True
    ),
    needs='fc7dropout_cudanet_out',
    provides='fc8_cudanet_out'
)
net.add_layer(
    core_layers.SoftmaxLayer(
        name='probs',
    ),
    needs='fc8_cudanet_out',
    provides='probs_cudanet_out'
)
net.finish()
visualize.draw_net_to_file(net, 'mine.png')
net.group_forward_backward(1)

print "###################################################################"
print "  Calling solver"
print "###################################################################"


MAXFUN = 500
solver = core_solvers.SGDSolver(
    lbfgs_args={'maxfun': MAXFUN, 'disp': 1},
    lr_policy='exp',
    base_lr=1,
    gamma=0.5)
solver.solve(net)
