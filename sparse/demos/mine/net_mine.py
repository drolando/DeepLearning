from sparse import base
from sparse.util import smalldata, visualize
from sparse.layers import core_layers, convolution, fillers, regularization
from sparse.opt import core_solvers
from sparse.layers.data.dataset import ImageDataLayer
import numpy as np
import copy
from matplotlib import pyplot

#images = np.load('/home/daniele/Documents/project/sparse-release/sparse/util/_data/images.npy')


"""lena = smalldata.lena()
image1 = classify(lena)
img = smalldata.get_image('cat.jpg')
image2 = classify(img)
images = [image1]
'''
    now the image is (10, 227, 227, 3)
    10 sub-images
    227 x 227 dimension
    3 rgb
'''
"""

net = base.Net()
'''
    NdarrayDataLayer -- takes a bunch of data and then emits them as Blobs.
    sources - list of images (10, 227, 227, 3)
    forward - puts these images in Blobs and sends them to the next layer
'''
"""
net.add_layer(
    core_layers.NdarrayDataLayer(
        name='input',
        sources=images
    ),
    group=1,
    provides='data'
)
"""
net.add_layer(
    ImageDataLayer(
        name='input-layer',
        train='../../util/_data/train.txt'
    ),
    group=1,
    provides=['data', 'labels']
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
        #filler=fillers.GaussianRandFiller(),
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
'''
    forward - (10, 28, 28, 96)
'''
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
"""
#---------------------------------------------------------
net.add_layer(
    core_layers.SparseFilteringLayer(
        name='sparse_filter_1'
    ),
    group=1,
    needs='rnorm1_cudanet_out',
    provides='sparse_out_1'
)
#---------------------------------------------------------
"""
'''
    2 groups by 128 kernels --> 256
    forward - (10, 28, 28, 256)
'''
net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv2',
        num_kernels=256,
        ksize=5,
        stride=1,
        mode='same',
        #filler=fillers.GaussianRandFiller(),
    ),
    group=2,
    needs='rnorm1_cudanet_out',
    provides='conv2_cudanet_out'
)
"""
net.add_layer(
    core_layers.GroupConvolutionLayer(
        name='conv2',
        stride=1,
        num_kernels=128,
        pad=2,
        ksize=5,
        group=2,
        mode='same',
        filler=fillers.GaussianRandFiller(),
        reg=regularization.L1Regularizer(weight=4)
    ),
    #needs='rnorm1_cudanet_out',
    group=2,
    needs='rnorm1_cudanet_out',
    provides='conv2_cudanet_out'
)"""

'''
    forward - (10, 28, 28, 256)
'''
net.add_layer(
    core_layers.ReLULayer(
        name='conv2_neuron',
        freeze=False
    ),
    group=2,
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
    group=2,
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
    group=2,
    needs='pool2_cudanet_out',
    provides='rnorm2_cudanet_out'
)
#------ here values are still in order of ~50
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
        #filler=fillers.GaussianRandFiller(),
    ),
    group=3,
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
    group=3,
    needs='conv3_cudanet_out',
    provides='conv3_neuron_cudanet_out'
)
'''
    2 groups by 192 kernels --> 384
    forward - (10, 14, 14, 384)
'''
net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv4',
        num_kernels=384,
        ksize=3,
        stride=1,
        mode='same',
        #filler=fillers.GaussianRandFiller(),
    ),
    group=4,
    needs='conv3_neuron_cudanet_out',
    provides='conv4_cudanet_out'
)
"""
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
    group=4,
    needs='conv3_neuron_cudanet_out',
    provides='conv4_cudanet_out'
)"""
net.add_layer(
    core_layers.ReLULayer(
        name='conv4_neuron',
        freeze=False
    ),
    group=4,
    needs='conv4_cudanet_out',
    provides='conv4_neuron_cudanet_out'
)
net.add_layer(
    core_layers.ConvolutionLayer(
        name='conv5',
        num_kernels=256,
        ksize=3,
        stride=1,
        mode='same',
        #filler=fillers.GaussianRandFiller(),
    ),
    group=5,
    needs='conv4_neuron_cudanet_out',
    provides='conv5_cudanet_out'
)
"""
net.add_layer(
    core_layers.GroupConvolutionLayer(
        name='conv5',
        stride=1,
        num_kernels=128,
        pad=1,
        ksize=3,
        group=2,
        filler=fillers.GaussianRandFiller()
    ),
    group=5,
    needs='conv4_neuron_cudanet_out',
    provides='conv5_cudanet_out'
)"""
net.add_layer(
    core_layers.ReLULayer(
        name='conv5_neuron',
        freeze=False
    ),
    group=5,
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
    group=5,
    needs='conv5_neuron_cudanet_out',
    provides='pool5_cudanet_out'
)
net.add_layer(
    core_layers.FlattenLayer(
        name='fc6_flatten',
    ),
    group=5,
    needs='pool5_cudanet_out',
    provides='_sparse_fc6_flatten_out'
)
net.add_layer(
    core_layers.InnerProductLayer(
        name='fc6',
        num_output=4096,
        has_bias=True,
        #filler=fillers.GaussianRandFiller(),
    ),
    group=6,
    needs='_sparse_fc6_flatten_out',
    provides='fc6_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='fc6_neuron',
        freeze=False
    ),
    group=6,
    needs='fc6_cudanet_out',
    provides='fc6_neuron_cudanet_out'
)
net.add_layer(
    core_layers.DropoutLayer(
        name='fc6dropout',
        ratio=0.500000,
    ),
    group=6,
    needs='fc6_neuron_cudanet_out',
    provides='fc6dropout_cudanet_out'
)
net.add_layer(
    core_layers.InnerProductLayer(
        name='fc7',
        num_output=4096,
        has_bias=True,
        #filler=fillers.GaussianRandFiller(),
    ),
    group=7,
    needs='fc6dropout_cudanet_out',
    provides='fc7_cudanet_out'
)
net.add_layer(
    core_layers.ReLULayer(
        name='fc7_neuron',
        freeze=False
    ),
    group=7,
    needs='fc7_cudanet_out',
    provides='fc7_neuron_cudanet_out'
)
net.add_layer(
    core_layers.DropoutLayer(
        name='fc7dropout',
        ratio=0.500000,
    ),
    group=7,
    needs='fc7_neuron_cudanet_out',
    provides='fc7dropout_cudanet_out'
)
net.add_layer(
    core_layers.InnerProductLayer(
        name='fc8',
        num_output=1000,
        has_bias=True,
        #filler=fillers.GaussianRandFiller()
    ),
    group=8,
    needs='fc7dropout_cudanet_out',
    provides='fc8_cudanet_out'
)
net.add_layer(
    core_layers.SoftmaxLayer(
        name='probs',
    ),
    group=8,
    needs='fc8_cudanet_out',
    provides='probs_cudanet_out'
)
net.add_layer(
    core_layers.KLDivergenceLossLayer(
        name='loss',
    ),
    group=8,
    needs=['probs_cudanet_out', 'labels']
)

net.finish()
old_kernels = copy.deepcopy(net.layers['conv1']._kernels._data)
visualize.draw_net_to_file(net, 'mine.png')

print "@#@#@#@#", net._input_blobs
print "@#@#@#@#", net._output_blobs

net.group_forward_backward(1)



print "###################################################################"
print "  Calling solver"
print "###################################################################"


MAXFUN = 500
solver = core_solvers.SGDSolver(
    lbfgs_args={'maxfun': MAXFUN, 'disp': 1},
    lr_policy='exp',
    base_lr=1,
    gamma=0.5,
    max_iter=100,
    disp=1)

#solver = core_solvers.LBFGSSolver(
#    lbfgs_args={'maxfun': MAXFUN, 'disp': 1})
solver.solve(net)

filters = net.layers['conv1'].param()[0].data()
_ = visualize.show_multiple(filters.T)
pyplot.show()
