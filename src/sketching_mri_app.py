# -*- coding: utf-8 -*-
"""This module contains an abstract class App for sketched iterative reconstruction,
and provides a few general Apps, including a sketched linear least squares App,
and a maximum eigenvalue estimation App.
"""
import numpy as np
import time


from sketching_app import SketchedLinearLeastSquares
# from builtins import None
import sigpy as sp
from sigpy.mri import linop
from sigpy.mri.app import _estimate_weights
from scipy.linalg import hadamard

class CoilSketching(SketchedLinearLeastSquares):

    def __init__(self, y, mps, reduced_ncoils, number_non_sketched_coils=None,
                 number_non_sketched_coils_init=None, sketch_type='Rademacher',
                 weights=None, coord=None, coil_batch_size=None, sketch_arrays=None,
                 device=sp.cpu_device, sketch_sigma=None, img_shape=None, **kwargs):

        self.img_shape = img_shape
        self.mps = mps
        self.mps_S = None
        self.mps_S_array = None
        self.reduced_ncoils = reduced_ncoils
        self.total_ncoils = mps.shape[0]
        self.number_non_sketched_coils = number_non_sketched_coils
        if self.number_non_sketched_coils is None:
            self.number_non_sketched_coils = self.reduced_ncoils - 1

        self.coil_batch_size = coil_batch_size
        if self.coil_batch_size is None:
            self.coil_batch_size = self.total_ncoils

        self.sketch_type = sketch_type
        self.sketch_arrays = sketch_arrays
        self.sketch_sigma = sketch_sigma

        weights = _estimate_weights(y, weights, coord)
        if weights is not None:
            y *= weights**0.5

        self.weights = weights
        self.coord = coord

        A = linop.Sense(mps, coord=coord, weights=weights,
                        coil_batch_size=coil_batch_size)

        super().__init__(A, y, device=device, **kwargs)

    # def _make_model_A(self):
    #     self.A_S = linop.Sense(self.mps, coord=self.coord, weights=self.weights,
    #                                 coil_batch_size=self.reduced_ncoils)
    #     return
    #
    # def _make_sketched_model_A(self):
    #
    #     mps_t, self.sigma_t  = self.make_sketched_model(self.mps, self.reduced_ncoils, self.sketch_type,
    #                                      nch_nsk=self.number_non_sketched_coils)
    #     self.A_S = linop.Sense(mps_t, coord=self.coord, weights=self.weights)
    #
    #     return

    def _load_sketched_model_A_S(self):
        nch_nsk = self.number_non_sketched_coils
        nch_sk = self.reduced_ncoils - nch_nsk
        isk1 = self.outer_iter * nch_sk
        isk2 = (self.outer_iter + 1) * nch_sk

        if self.mps_S is None:
            self.mps_S = self.device.xp.zeros([self.reduced_ncoils]+list(self.img_shape), dtype=self.mps.dtype)

        self.mps_S[:nch_nsk,...] = sp.to_device(self.mps[:nch_nsk, ...], device=self.device)
        self.mps_S[nch_nsk:,...] = sp.to_device(self.mps_S_array[isk1:isk2, ...], device=self.device)
        self.A_S = linop.Sense(self.mps_S, coord=self.coord, weights=self.weights)
        return

    def _make_sketched_models_A_S(self):
        mps = self.mps
        nch = self.reduced_ncoils
        nc = self.total_ncoils
        nch_nsk = self.number_non_sketched_coils
        sketch_type = self.sketch_type
        sigma = self.sketch_sigma
        img_shape = self.img_shape

        if nch_nsk == -1:
            nch_nsk = nch

        nch_sk = nch - nch_nsk
        nc_sk = nc - nch_nsk
        dim_c = nc-nch_nsk
        dim_ch = nch-nch_nsk
        sk_size = [dim_c] + [1]*len(img_shape) + [self.max_outer_iter * dim_ch]

        if nch_sk > 0:
            if sigma is None:
                if sketch_type == 'Gaussian':
                    sigma = np.random.normal(0,1, [dim_c, self.max_outer_iter * dim_ch])/np.sqrt(nch_sk)

                elif sketch_type == 'Rademacher':
                    sigma = (np.random.randint(0, 2, size=[dim_c, self.max_outer_iter * dim_ch])*2 - 1)/np.sqrt(nch_sk)

            # Getting sketched coils
            sigma = sigma.astype(mps.dtype)
            sigma1 = np.reshape(sigma, sk_size)
            mps_sk = np.sum(np.expand_dims(mps[nch_nsk:,...],-1) * sigma1, 0)
            mps_sk = np.moveaxis(mps_sk, -1, 0)

        self.sketch_sigma = sigma
        self.mps_S_array = mps_sk
        return
    # def make_sketched_model(self, mps, nch, sketch_type, nch_nsk=None, y=None, sigma=None):
    #
    #
    #     mps_shape = mps.shape
    #     nc = mps_shape[0]
    #     ishape = mps_shape[1:]
    #
    #     if nch_nsk is None:
    #         nch_nsk = nch - 1
    #     elif nch_nsk == -1:
    #         nch_nsk = nch
    #
    #     nch_sk = nch - nch_nsk
    #     nc_sk = nc - nch_nsk
    #
    #     # #Sketching matrix
    #     sk_size = [nc-nch_nsk] + [1]*len(ishape) + [nch-nch_nsk]
    #      #Sketching matrix
    #     # sk_size = [nc]
    #     # sk_size.extend([1]*len(ishape))
    #     # sk_size.append(nch)
    #
    #     #Sensitivity maps
    #     mps_t = np.zeros([nch]+list(ishape), dtype=mps.dtype)
    #     mps_t[:nch_nsk, ...] = mps[:nch_nsk, ...]
    #
    #     # sigma = np.zeros([nc,nch], dtype=mps.dtype)
    #     # sigma[range(nch_nsk),range(nch_nsk)] = 1
    #
    #     if nch_sk > 0:
    #         if sigma is None:
    #             if sketch_type == 'Gaussian':
    #                 # Gaussian sketch
    #                 sigma = np.random.normal(0,1, [nc-nch_nsk,nch-nch_nsk])/np.sqrt(nch_sk)
    #                 # sigma = np.random.normal(0,1, [nc,nch])/np.sqrt(nch)
    #
    #             elif sketch_type == 'Rademacher':
    #                 sigma = (np.random.randint(0, 2, size=[nc-nch_nsk,nch-nch_nsk])*2 - 1)/np.sqrt(nch_sk)
    #
    #         sigma = sigma.astype(mps.dtype)
    #         sigma1 = np.reshape(sigma, sk_size)
    #         mps_sk = np.sum(np.expand_dims(mps[nch_nsk:,...],-1) * sigma1, 0)
    #         mps_sk = np.moveaxis(mps_sk, -1, 0)
    #         mps_t[nch_nsk:, ...] = mps_sk
    #
    #         # print(sigma)
    #         # sigma = np.reshape(sigma, sk_size)
    #         # mps_t = np.sum(np.expand_dims(mps,-1) * sigma, 0)
    #         # mps_t = np.moveaxis(mps_t, -1, 0)
    #
    #     return (mps_t, sigma)

    def _make_initial_sketched_problem(self):

        self.mps_S = sp.to_device(self.mps[:self.reduced_ncoils, ...], device=self.device)
        self.A_S = linop.Sense(self.mps_S, coord=self.coord, weights=self.weights)
        self.y_S = sp.to_device(self.y[:self.reduced_ncoils,...], device=self.device)

        return

    def _get_AHy(self):
        coil_batch_size = min(self.reduced_ncoils, self.coil_batch_size)
        self.A_S = linop.Sense(self.mps, coord=self.coord, weights=self.weights,
                    coil_batch_size=coil_batch_size)
        self.AHy = self.A_S.H(sp.to_device(self.y, self.device))
        return

    def _get_true_gradient(self):
        #Memory efficient calculation of true gradient
        num_coil_batches = (self.total_ncoils + self.reduced_ncoils - 1) // self.reduced_ncoils

        with self.device:
            self.d = -self.AHy

            for c in range(num_coil_batches):
                self.mps_S = sp.to_device(self.mps[c*self.reduced_ncoils:((c+1)*self.reduced_ncoils)], self.device)
                self.A_S = linop.Sense(self.mps_S, coord=self.coord, weights=self.weights)
                self.d += self.A_S.N(self.x)

        return

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

        super().__init__(y, mps, reduced_ncoils, proxg=proxg, g=g,
                                        img_shape=img_shape, **kwargs)


class SketchedTotalVariationRecon(CoilSketching):
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

        super().__init__(y, mps, reduced_ncoils, proxg=proxg, g=g, G=G,
                                                img_shape=img_shape, **kwargs)


class SketchedSenseRecon(CoilSketching):
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
        super().__init__(y, mps, reduced_ncoils, lamda=lamda,
                                        img_shape=img_shape, **kwargs)
