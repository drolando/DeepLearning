"""Implements the sparsefiltering layer."""

from sparse import base
from sparse.util import logexp
import numpy as np
import numexpr

class SparseFilteringLayer(base.Layer):
    """A layer that implements the sparse filtering operation."""

    def __init__(self, **kwargs):
        base.Layer.__init__(self, **kwargs)
    
    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and top_data
        top[0]._data = bottom[0]._data
        self._top = top[0]
        print "---------- ------------------ top ", top[0]._data.shape

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        print "sparse_backward"
        top = [self._top]
        if top[0].has_diff() == False:
            top[0].init_diff()
        if propagate_down:
            top_diff = top[0].diff()
            bottom_diff = bottom[0].init_diff(setzero=False)
            top_diff.shape = (top_diff.size / top_diff.shape[-1], top_diff.shape[-1])
            bottom_diff.shape = top_diff.shape
            bottom_diff[:] = top_diff
            bottom_diff -= (top_diff.sum(1) / top_diff.shape[-1])[:, np.newaxis]
        return 0

    def update(self):
        """SparseFiltering has nothing to update."""
        pass
