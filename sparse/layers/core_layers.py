# pylint: disable=W0611
"""Imports commonly used layers."""

# Utility layers
from sparse.base import SplitLayer

# Data Layers
from sparse.layers.data.ndarraydata import NdarrayDataLayer
from sparse.layers.data.cifar import CIFARDataLayer
from sparse.layers.data.mnist import MNISTDataLayer
from sparse.layers.sampler import (BasicMinibatchLayer,
                                  RandomPatchLayer)
#from sparse.layers.puffsampler import PuffSamplerLayer

# Computation Layers
from sparse.layers.convolution import ConvolutionLayer
from sparse.layers.group_convolution import GroupConvolutionLayer
#from sparse.layers.deconvolution import DeconvolutionLayer
from sparse.layers.dropout import DropoutLayer
from sparse.layers.flatten import FlattenLayer
#from sparse.layers.identity import IdentityLayer
from sparse.layers.im2col import Im2colLayer
from sparse.layers.innerproduct import InnerProductLayer
from sparse.layers.loss import (SquaredLossLayer,
                               LogisticLossLayer,
                               MultinomialLogisticLossLayer,
                               KLDivergenceLossLayer,
                               AutoencoderLossLayer)
from sparse.layers.normalize import (MeanNormalizeLayer,
                                    ResponseNormalizeLayer,
                                    LocalResponseNormalizeLayer)
#from sparse.layers.padding import PaddingLayer
from sparse.layers.pooling import PoolingLayer
from sparse.layers.relu import ReLULayer
from sparse.layers.sigmoid import SigmoidLayer
from sparse.layers.softmax import SoftmaxLayer

from sparse.layers.sparsefiltering import SparseFilteringLayer
