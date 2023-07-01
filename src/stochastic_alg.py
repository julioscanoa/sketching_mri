# -*- coding: utf-8 -*-
"""This module contains an abstract class App for sketched iterative reconstruction,
and provides a few general Apps, including a sketched linear least squares App,
and a maximum eigenvalue estimation App.
"""
import numpy as np
import sigpy as sp
import time

from tqdm.auto import tqdm
from sigpy import backend, linop, prox, util
from sigpy.alg import GradientMethod, PrimalDualHybridGradient

# from utils import TotalVariationRecon_PDHGmod

class StochasticGradient(GradientMethod):

    def __init__(self, x, alpha, beta=1.0, device=sp.cpu_device, proxg=None,
                    accelerate=False, max_iter=5, tol=0):

        self.x = x
        self.alpha = alpha
        self.alpha0 = alpha
        self.beta = beta
        self.proxg = proxg
        self.device = device
        self.accelerate = accelerate
        self.max_iter = max_iter
        self.true_gradf = None
        self.sub_gradf = None
        self.tol = tol
        self.iter = 0
        self.resid = np.infty
        # self.gradf = self._get_stochastic_grad

        with self.device:
            if self.accelerate:
                self.z = self.x.copy()
                self.t = 1.0

    # def _get_stochastic_grad(self, x):
    #     output = self.sub_gradf(x)
    #     return output

    def _update_stepsize(self):
        self.alpha = self.alpha0*max(self.beta**self.iter, 1.0/self.iter)
        return

    def _done(self):
        return (self.iter >= self.max_iter) or self.resid <= self.tol
