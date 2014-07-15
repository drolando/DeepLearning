# coding=utf-8
# This 2 lines allow the network to be executed on a remote server without graphic interface.
import matplotlib
matplotlib.use('Agg')

from sparse import base
from sparse.util import smalldata, visualize
from sparse.layers import core_layers, convolution, fillers, regularization, sparsenet, sparsefiltering
from sparse.opt import core_solvers
from sparse.layers.data.dataset import ImageDataLayer
import numpy as np
from matplotlib import pyplot
import sys, time, datetime
from sparse.util.tsne_python import tsne

net = sparsenet.SparseNet()

net.finish()
visualize.draw_net_to_file(net, 'network.png')