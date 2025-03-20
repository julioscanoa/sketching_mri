# -*- coding: utf-8 -*-
"""This module contains an abstract class App for sketched iterative reconstruction,
and provides a few general Apps, including a sketched linear least squares App,
and a maximum eigenvalue estimation App.
"""
import numpy as np
import time
import random

# from builtins import None
import sigpy as sp
from sigpy.mri import linop
from sigpy.mri.app import _estimate_weights
from scipy.linalg import hadamard
from stochastic_app import StochasticLinearLeastSquares

class StochasticCoilRecon(StochasticLinearLeastSquares):

    def __init__(self, y, mps, reduced_ncoils, fixed_ncoils=0,
                 weights=None, coord=None, coil_batch_size=None,
                 device=sp.cpu_device, **kwargs):

        self.mps = mps
        self.total_ncoils = mps.shape[0]
        self.reduced_ncoils = reduced_ncoils
        self.fixed_ncoils = fixed_ncoils
        self.coil_list = np.arange(fixed_ncoils, self.total_ncoils)

        self.coil_batch_size = coil_batch_size
        self.weights = weights
        self.coord = coord
        self.coil_list = np.arange(self.fixed_ncoils, self.total_ncoils)

        weights = _estimate_weights(y, weights, coord)
        if weights is not None:
            y *= weights**0.5

        A = linop.Sense(mps, coord=coord, weights=weights,
                        coil_batch_size=coil_batch_size)

        super().__init__(A, y, device=device, **kwargs)


    def _get_AHy(self):
        self.A_S = linop.Sense(self.mps, coord=self.coord, weights=self.weights,
                    coil_batch_size=self.reduced_ncoils)
        self.AHy = self.A_S.H(sp.to_device(self.y, self.device))
        return


    def _get_subsampled_problem(self):

        ind_c = self._get_random_index()
        A_S = linop.Sense(self.mps[ind_c,...], coord=self.coord, weights=self.weights)
        y_S = sp.to_device(self.y[ind_c,...], device=self.device)
        coeff = self.total_ncoils/self.reduced_ncoils

        return coeff, A_S, y_S

    def _get_hessian_for_alpha(self):
        # A_S = linop.Sense(self.mps[0,...], coord=self.coord, weights=self.weights)
        # coeff = self.total_ncoils/self.reduced_ncoils
        A_S = linop.Sense(self.mps, coord=self.coord, weights=self.weights,
                        coil_batch_size=self.reduced_ncoils)

        return A_S.H * A_S

    def _get_random_index(self):
        nc = self.total_ncoils
        ncs = self.reduced_ncoils-self.fixed_ncoils
        ind_c = np.zeros([0],dtype=int)

        while len(ind_c) < ncs:
            size_c = min(ncs, len(self.coil_list), ncs-len(ind_c))
            if len(self.coil_list) == 0:
                self.coil_list = np.arange(self.fixed_ncoils, self.total_ncoils)
            ind0 = np.random.choice(len(self.coil_list), size=size_c, replace=False)
            ind_c = np.concatenate((ind_c,self.coil_list[ind0]))
            self.coil_list = np.delete(self.coil_list, ind0)
        ind_c = np.concatenate((np.arange(self.fixed_ncoils), ind_c))

        return ind_c

class StochasticL1WaveletRecon(StochasticCoilRecon):
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

    def __init__(self, y, mps, lamda, reduced_ncoils,
                 wave_name='db4', **kwargs):

        img_shape = mps.shape[1:]
        W = sp.linop.Wavelet(img_shape, wave_name=wave_name)
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, lamda), W)

        def g(input):
            device = sp.get_device(input)
            xp = device.xp
            with device:
                return lamda * xp.sum(xp.abs(W(input))).item()

        super().__init__(y, mps, reduced_ncoils, proxg=proxg, g=g, **kwargs)


class StochasticTotalVariationRecon(StochasticCoilRecon):
    r"""Total variation regularized reconstruction.

    Solves the following problem efficiently using Coil Sketching:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| G x \|_1

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, G is the gradient operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Block, K. T., Uecker, M., & Frahm, J. (2007).
        Undersampled radial MRI with multiple coils.
        Iterative image reconstruction using a total variation constraint.
        Magnetic Resonance in Medicine, 57(6), 1086-1098.

    """

    def __init__(self, y, mps, lamda, reduced_ncoils,
                 **kwargs):

        img_shape = mps.shape[1:]
        G = sp.linop.FiniteDifference(img_shape)
        proxg = sp.prox.L1Reg(G.oshape, lamda)

        def g(x):
            device = sp.get_device(x)
            xp = device.xp
            with device:
                return lamda * xp.sum(xp.abs(x)).item()

        super().__init__(y, mps, reduced_ncoils, proxg=proxg, g=g, G=G, **kwargs)


class StochasticSenseRecon(StochasticCoilRecon):
    r"""SENSE Reconstruction.

    Solves the following problem efficiently using Coil Sketching:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 +
        \frac{\lambda}{2} \| x \|_2^2

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        tseg (None or Dictionary): parameters for time-segmented off-resonance
            correction. Parameters are 'b0' (array), 'dt' (float),
            'lseg' (int), and 'n_bins' (int). Lseg is the number of
            time segments used, and n_bins is the number of histogram bins.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
            Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Pruessmann, K. P., Weiger, M., Scheidegger, M. B., & Boesiger, P.
        (1999).
        SENSE: sensitivity encoding for fast MRI.
        Magnetic resonance in medicine, 42(5), 952-962.

        Pruessmann, K. P., Weiger, M., Bornert, P., & Boesiger, P. (2001).
        Advances in sensitivity encoding with arbitrary k-space trajectories.
        Magnetic resonance in medicine, 46(4), 638-651.

    """

    def __init__(self, y, mps, lamda, reduced_ncoils,
                 **kwargs):
        super().__init__(y, mps, reduced_ncoils, lamda=lamda, **kwargs)
