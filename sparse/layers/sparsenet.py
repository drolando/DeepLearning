from sparse import base
from sparse.layers.data import dataset
from sparse.opt import core_solvers
from sparse.layers import core_layers, sparsefiltering
from sparse.util.timer import Timer
import numpy as np
import logging
#from sklearn.feature_extraction import image

#from scipy.io import loadmat
#from sparse_filtering import SparseFiltering

PATCHES_PER_IMAGE = 50
RANDOM_SEED = 31
MAX_ITER = 100
MAX_DEPTH = 10
DISP = 50


class SparseNet(base.Net):
    def __init__(self, **kwargs):
        base.Net.__init__(self, **kwargs)
        self.input_layer = None
        self.conv_layers = []
        self.logger = logging.getLogger("CSHEL_parserlogger")
        self.logger.setLevel(logging.DEBUG)


    def sparse_filtering(self):
        self.logger.info('Sparse Filtering: started.')
        count = 0
        for layer in self.conv_layers:
            if count in [0, 1]:#, 2, 3, 4, 5]:
                self.process_layer(layer, count)
            count += 1


    # returns just 1 image
    def feed_forward(self, depth):
        last = None
        for _, layer, bottom, top in self._forward_order:
            if layer._net_group <= depth:
                layer.forward(bottom, top)
                last = top
            else:
                break
        if last == None:
            raise ValueError('No layer processed. Set group property to each layer!')
        return last[0]._data


    def process_layer(self, layer, depth):
        self.logger.info('Sparse Filtering: processing layer %s.'%(layer.name,))
        kernel_size = None
        self.count = 0

        if type(layer) == core_layers.GroupConvolutionLayer:
            ksize = layer._conv_layers[0]._ksize
        else:
            ksize = layer._ksize

        # extracts patches from all images
        timer = Timer()
        for i in xrange(self.input_layer._num_images):
            input_data = self.feed_forward(depth)
            if type(layer) == core_layers.GroupConvolutionLayer:
                input_channels = input_data.shape[-1]# / layer._group
            else:
                input_channels = input_data.shape[-1]

            if kernel_size is None:
                kernel_size = (ksize*ksize*input_channels, self.input_layer._num_images * PATCHES_PER_IMAGE)
                self.patches = np.zeros(kernel_size, dtype=np.float32)

            self.last_img = input_data
            assert not np.isnan(input_data.sum())
            layer.extract_patches(input_data, self, PATCHES_PER_IMAGE)
        self.logger.info('Sparse Filtering: feed forward time %s.'%(timer.total()))

        '''if depth == 0:
            data = loadmat('patches.mat')['data']
            self.patches -= self.patches.mean(axis=0)'''

        '''if depth in [3, 4, 5]:
            MAX_ITER = 400
        else:
            MAX_ITER = 1000'''
        # normalizes patches
        res = sparsefiltering.sparseFiltering(layer._num_kernels, self.patches, max_iter=MAX_ITER, disp=DISP, net=self, layer=layer)

        kernels = []
        if type(layer) == core_layers.GroupConvolutionLayer:
            for l in layer._conv_layers:
                kernels.append(l._kernels)
                if not l._kernels.has_data():
                    data = l._kernels.init_data(
                        (kernel_size[0], l._num_kernels),
                        np.float32)
        else:
            kernels.append(layer._kernels)
            if not layer._kernels.has_data():
                data = layer._kernels.init_data(
                    (kernel_size[0], layer._num_kernels),
                    np.float32)

        for k in kernels:
            k._data[:] = res.T
        #data[:] = res.T
        #data[:] = self.sp.w_.T


    def extract_patches(self, data, ksize):
        cnt = 0

        for i in xrange(PATCHES_PER_IMAGE): # 10
            vect = image.extract_patches_2d(data[cnt], (ksize, ksize), 1, RANDOM_SEED)
            self.add_patch(vect[0])
            cnt  = (cnt + 1) % data.shape[0]


    def add_patch(self, vect):
        (self.patches.T)[self.count] = vect.flatten().T
        self.count += 1


    def add_layer(self, layer, needs=None, provides=None, group=-1):
        super(SparseNet, self).add_layer(layer, needs, provides, group)
        if type(layer) == dataset.ImageDataLayer:
            #input layer
            self.input_layer = layer
        elif type(layer) == core_layers.ConvolutionLayer or type(layer) == core_layers.GroupConvolutionLayer:
            #convolutional layers
            self.conv_layers.append(layer)



