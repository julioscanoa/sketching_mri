# -*- coding: utf-8 -*-
"""This module contains an abstract class App for sketched iterative reconstruction,
and provides a few general Apps, including a sketched linear least squares App,
and a maximum eigenvalue estimation App.
"""
import numpy as np
import time
import random

from sketching_app_v2 import SketchedLinearLeastSquares
# from builtins import None
import sigpy as sp
from sigpy.mri import linop
from sigpy.mri.app import _estimate_weights

class CoilSketching(SketchedLinearLeastSquares):

    def __init__(self, y, mps, number_of_coils, mps_S=None, 
                 number_of_fixed_coils=None, sigma=None,
                 weights=None, coord=None, coil_batch_size=None,
                 device=sp.cpu_device, **kwargs):

        self.mps = mps
        self.mps_S = mps_S
        self.total_number_of_coils = mps.shape[0]
        self.number_of_coils = number_of_coils
        self.number_of_fixed_coils = number_of_fixed_coils

        if self.number_of_fixed_coils is None:
            self.number_of_fixed_coils = self.number_of_coils - 1

        if coil_batch_size is None:
            self.coil_batch_size = self.number_of_coils

        self.coil_batch_size = coil_batch_size
        self.weights = weights
        self.coord = coord

        if self.mps_S is None:
            self._precalc_sketched_coils()

        weights = _estimate_weights(y, weights, coord)
        if weights is not None:
            y *= weights**0.5

        A = linop.Sense(mps, coord=coord, weights=weights,
                        coil_batch_size=1)

        super().__init__(A, y, device=device, **kwargs)


    def _get_Ahy(self):
        self.A_S = linop.Sense(self.mps, coord=self.coord, weights=self.weights,
                    coil_batch_size=self.number_of_coils)
        self.Ahy = self.A_S.H(sp.to_device(self.y, self.device))
        return

    def _get_true_gradient(self):
        #Memory efficient calculation of true gradient
        num_coils0 = len(self.mps)
        num_coil_batches = (num_coils0 + self.number_of_coils - 1) // self.number_of_coils

        with self.device:
            self.d = -self.Ahy
            for c in range(num_coil_batches):
                self.A_S = linop.Sense(self.mps[c*self.number_of_coils:((c+1)*self.number_of_coils)],
                                coord=self.coord, weights=self.weights)
                self.d += self.A_S.N(self.x)

        return self.d

    def _precalc_sketched_coils(self):
        self.mps_S = np.zeros((self.max_outer_iter, self.number_of_coils, *self.mps.shape[1:]), dtype=self.mps.dtype)
        self.mps_S[:,:self.number_of_fixed_coils] = self.mps[:self.number_of_fixed_coils]

        if self.sigma is None:
            self.sigma = np.random.randint(0, 2, (self.max_outer_iter, self.number_of_coils - self.number_of_fixed_coils, \
                                                  self.total_number_of_coils - self.number_of_fixed_coils)) * 2 -1
        
        num_dims = len(self.mps.shape[1:])
        self.sigma = np.reshape(self.sigma, self.sigma.shape + (1,) * num_dims)
        
        # Once in cpu
        self.mps_S[:,self.number_of_fixed_coils:] = np.sum(self.sigma * self.mps[None, None, self.number_of_fixed_coils:, ...], axis=2)
        
        # # Memory efficient
        num_c_loops = (self.total_number_of_coils - self.number_of_fixed_coils) // self.number_of_coils
        for ic in range(num_c_loops):
            self.sigma_dev = sp.to_device(self.sigma[:, ic*self.number_of_coils:(ic+1)*self.number_of_coils], self.device)
            self.mps_dev = sp.to_device(self.mps[self.number_of_fixed_coils + ic*self.number_of_coils:(ic+1)*self.number_of_coils], self.device)
            self.mps_S[:,self.number_of_fixed_coils] += self.device.xp.sum(self.sigma_dev * self.mps_dev[None, ...], axis=2)
        return

    def _get_sketched_problem(self, ind_sketch):
        A_S = linop.Sense(self.mps_S[ind_sketch,...], coord=self.coord, weights=self.weights)
        return A_S


class SketchedL1WaveletRecon(CoilSketching):
    r"""L1 Wavelet regularized reconstruction.

    Solves the following problem efficiently using Coil Sketching:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| W x \|_1

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, W is the wavelet operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        wave_name (str): wavelet name.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Lustig, M., Donoho, D., & Pauly, J. M. (2007).
        Sparse MRI: The application of compressed sensing for rapid MR imaging.
        Magnetic Resonance in Medicine, 58(6), 1082-1195.

    """

    def __init__(self, y, mps, lamda, number_of_coils,
                 wave_name='db4', **kwargs):

        img_shape = mps.shape[1:]
        W = sp.linop.Wavelet(img_shape, wave_name=wave_name)
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, lamda), W)

        def g(input):
            device = sp.get_device(input)
            xp = device.xp
            with device:
                return lamda * xp.sum(xp.abs(W(input))).item()

        super().__init__(y, mps, number_of_coils, proxg=proxg, g=g, **kwargs)


class SketchCoils():

    def __init__(self, mps_ker, img_shape, number_of_coils, max_outer_iter, 
                 number_of_fixed_coils=None, sigma=None, seed=0):

        self.mps_ker = mps_ker
        self.img_shape = img_shape
        self.device = sp.get_device(mps_ker)
        self.number_of_coils = number_of_coils
        self.max_outer_iter = max_outer_iter
        self.number_of_fixed_coils = number_of_fixed_coils
        if self.number_of_fixed_coils is None:
            self.number_of_fixed_coils = self.number_of_coils - 1
        self.sigma = sigma

        self.mps_S = None
        self.mps_Sk = None
        self.seed = seed

        np.random.seed(self.seed)

        return
    
    def precalc_sketched_coils(self):

        """
        Pre-calculate sketched coils for Coil Sketching.
        Args:
            mps (array): sensitivity maps. Shape (number_of_coils, *img_shape).
            number_of_coils (int): number of coils to sketch.
            max_outer_iter (int): maximum number of outer iterations.
            number_of_fixed_coils (int): number of fixed coils.
            sigma (array): random numbers for sketching.
        Returns:
            mps_S (array): sketched sensitivity maps. Shape (max_outer_iter, number_of_coils, *img_shape).S
        """
        xp = self.device.xp

        total_number_of_coils = self.mps_ker.shape[0]

        with self.device:
            if self.sigma is None:
                self.sigma = np.random.randint(0, 2, (self.max_outer_iter, self.number_of_coils - self.number_of_fixed_coils, \
                                                        total_number_of_coils - self.number_of_fixed_coils), dtype=np.int32) * 2 -1
                self.sigma = self.sigma.astype(self.mps.dtype)
                self.sigma = sp.to_device(self.sigma, self.device)
            
            
            num_dims = len(self.mps_ker.shape[1:])
            sigma = xp.reshape(self.sigma, self.sigma.shape + (1,) * num_dims)
            
            self.mps_Sk = xp.sum(sigma * self.mps_ker[None, None, self.number_of_fixed_coils:, ...], axis=2)
        
        self.mps_Sk = self.resize_coils(self.mps_Sk, self.img_shape)
        return 

    def resize_coils(self, mps_ker, img_shape):
        """
        Resize sensitivity maps to match image shape.
        Args:
            mps_ker (array): sensitivity maps. Shape (number_of_coils, *img_shape).
            img_shape (tuple): image shape.
        Returns:
            mps (array): resized sensitivity maps. Shape (number_of_coils, *img_shape).
        """

        with self.device:
            mps = np.zeros((mps_ker.shape[0], mps_ker.shape[1], *img_shape), dtype=mps_ker.dtype)
            for i in range(mps_ker.shape[0]):
                for j in range(mps_ker.shape[1]):
                    sp.copyto(mps[i,j], sp.ifft(sp.resize(mps_ker[i,j], img_shape)))

        return mps

    
    def run(self):
        
        self.mps = self.resize_coils(self.mps_ker[:self.number_of_fixed_coils][None,...], self.img_shape)
        self.mps = np.squeeze(self.mps, axis=0)
        self.precalc_sketched_coils()

        self.mps_S = np.zeros((self.max_outer_iter, self.number_of_coils, *self.img_shape), dtype=self.mps.dtype)
        self.mps_S[:,:self.number_of_fixed_coils] = self.mps
        self.mps_S[:,self.number_of_fixed_coils:] = self.mps_Sk

        # Normalization
        rss = np.sum(np.abs(self.mps**2), axis=0)[None, ...]
        rss = rss + np.sum(np.abs(self.mps_Sk)**2, axis=1)
        rss = np.sqrt(rss)[:, None, ...]
        self.mps_S = self.mps_S / rss

        return self.mps_S.astype(self.mps.dtype)
