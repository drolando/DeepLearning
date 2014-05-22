from sparse import base
from sparse.layers import core_layers
from sparse.layers.data import dataset
from sparse.opt import core_solvers
from sparse.layers import sparsefiltering
from sparse.util.timer import Timer
import numpy as np
import logging
from sklearn.feature_extraction import image

PATCHES_PER_IMAGE = 10
RANDOM_SEED = 31

class SparseNet(base.Net):
	def __init__(self, **kwargs):
		base.Net.__init__(self, **kwargs)
		self.count = 0
		self.input_layer = None
		self.conv_layers = []
		self.logger = logging.getLogger("CSHEL_parserlogger")
		self.logger.setLevel(logging.DEBUG)


	def sparse_filtering(self):
		self.logger.info('Sparse Filtering: started.')
		count = 0
		for layer in self.conv_layers:
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

		# extracts patches from all images
		timer = Timer()
		for i in xrange(self.input_layer._num_images):
			input_data = self.feed_forward(depth)
			if kernel_size == None:
				kernel_size = (layer._ksize*layer._ksize*input_data.shape[-1], self.input_layer._num_images * PATCHES_PER_IMAGE)
				self.patches = np.zeros(kernel_size)
			self.extract_patches(input_data, layer._ksize)
		self.logger.info('Sparse Filtering: feed forward time %s.'%(timer.total()))

		# normalizes patches
		res = sparsefiltering.sparseFiltering(layer._num_kernels, self.patches)
		print res.T.shape
		if not layer._kernels.has_data():
			data = layer._kernels.init_data(
				(kernel_size[0], layer._num_kernels),
				res.dtype)
			data[:] = res


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
		elif type(layer) == core_layers.ConvolutionLayer:
			#convolutional layers
			self.conv_layers.append(layer)



