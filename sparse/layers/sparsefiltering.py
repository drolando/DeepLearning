"""Implements the sparsefiltering layer."""

from sparse import base
from sparse.util import logexp
import numpy as np
import numexpr

class SparseFilteringLayer(base.Layer):
    """A layer that implements the sparse filtering operation."""

    def __init__(self, **kwargs):
        base.Layer.__init__(self, **kwargs)
        self._Fhat = None

    
    def forward(self, bottom, top):
        """Computes the forward pass."""
        '''
            bottom = [(10000, 25)] --input
            top = [()] --output
        '''
        
        data = bottom[0].data()
        diff = bottom[0].init_diff()
        #data_mean = data.mean(axis=0)
        # we clip it to avoid overflow
        #np.clip(data_mean, np.finfo(data_mean.dtype).eps,
        #        1. - np.finfo(data_mean.dtype).eps,
        #        out=data_mean)
        #neg_data_mean = 1. - data_mean

        if data.shape != (10000, 25):
            self._loss = 0.
            return
        
        Fs = np.sqrt(data**2 + 1e-8)
        NFs, L2Fs = self.l2row(Fs)
        #NFs = matrix normalized by rows
        #L2Fs = squared sum over rows
        Fhat, L2Fn = self.l2row(NFs.T)
        #print (Fhat)[0]

        #self._loss = Fhat.sum()

        self._loss = (data - Fhat.T).sum()
        

        if len(top) > 0:
            output = top[0].init_data(data.shape, data.dtype, setdata=False)
            output[:] = Fhat.T            

        '''top[0]._data = Fhat.T

        top[0]._L2Fs = L2Fs
        top[0]._L2Fn = L2Fn
        top[0]._NFs= NFs'''

    def backward(self, bottom, top, propagate_down):
        '''
            bottom = [(10000, 25)] --output?
            top = [(10000, 25)] --input?
        '''
        #data = bottom[0].data()
        #Fhat = top[0]._data.T
        #bottom[0]._data = top[0]._data
        #Fs = np.sqrt(data**2 + 1e-8)

        '''DeltaW = self.l2rowg(top[0]._NFs.T, Fhat, top[0]._L2Fn, np.ones(Fhat.shape))
        #np.ones returns a matrix of all 1
        DeltaW = self.l2rowg(data, top[0]._NFs, top[0]._L2Fs, DeltaW.T)
        DeltaW = (DeltaW*(data/Fs)).dot(data.T)
        '''
        return self._loss

    def update(self):
        """SparseFiltering has nothing to update."""
        pass

    def l2row(self, X):
        """
        L2 normalize X by rows. We also use this to normalize by column with l2row(X.T)
        """
        N = np.sqrt((X**2).sum(axis=1)+1e-8)
        print N.shape
        #sum = Sum of array elements over a given axis.
        #axis = 1 means sum over rows
        #N.shape = (256,)
        Y = (X.T/N).T
        #Y is now normalized in some way...
        return Y,N


    def l2rowg(self, X,Y,N,D):
        """
        Backpropagate through Normalization.

        Parameters
        ----------

        X = Raw (possibly centered) data.
        Y = Row normalized data.
        N = Norms of rows.
        D = Deltas of previous layer. Used to compute gradient.

        Returns
        -------

        L2 normalized gradient.
        """
        return (D.T/N - Y.T * (D*X).sum(axis=1) / N**2).T
    