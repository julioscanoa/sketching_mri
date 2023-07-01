# -*- coding: utf-8 -*-
"""This module contains an abstract class App for sketched iterative reconstruction,
and provides a few general Apps, including a sketched linear least squares App,
and a maximum eigenvalue estimation App.
"""
import numpy as np
import sigpy as sp
import time
import random

from tqdm.auto import tqdm
from sigpy import backend, linop, prox, util
from sigpy.alg import (PowerMethod, GradientMethod)
from stochastic_alg import StochasticGradient

from sigpy.app import (App, MaxEig)
import scipy.io as sio
import cupy as cp
# from utils import TotalVariationRecon_PDHGmod

class StochasticLinearLeastSquares(App):
    def __init__(self, A, y, x=None,
                    proxg=None, lamda=0, G=None, g=None, z=None,
                    alpha=None, beta=1.0, tau=None, sigma=None, max_iter=100,
                    max_outer_iter=20, max_inner_iter=5, max_power_iter=30,
                    accelerate=True, rho=1, max_cg_iter=10, tol=0,
                    save_objective_values=False,
                    device=sp.cpu_device, solver=None,
                    show_pbar=True, leave_pbar=True,
                    seed=None, max_resid=np.infty,
                    save_images=None, path_save_images=None):

        self.A = A
        self.x = x
        self.y = y
        self.d = None
        self.proxg = proxg
        self.lamda = lamda
        self.G = G
        self.g = g
        self.z = z
        self.solver = solver

        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.sigma = sigma
        self.max_resid = max_resid

        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.max_power_iter = max_power_iter
        self.max_cg_iter = max_cg_iter
        self.max_iter = max_iter
        self.accelerate = accelerate

        self.seed = seed
        self.device = device
        self.objective_values = None
        self.time = None

        self.tol = tol
        self.show_pbar = show_pbar
        self.leave_pbar = leave_pbar
        self.save_objective_values = save_objective_values

        # random.seed(self.seed)
        np.random.seed(self.seed)
        if self.x is None:
            with self.device:
                self.x = self.device.xp.zeros(A.ishape, dtype=y.dtype)
        else:
            self.x = sp.to_device(self.x, device=self.device)

        # self._get_Ahy()
        self._get_alg()
        # self.y = sp.to_device(self.y, device=self.device)

        if self.save_objective_values:
            self.y = sp.to_device(self.y, device=self.device)
            self.objective_values = [self.objective()]
        else:
            self.A = None

        super().__init__(self.alg, show_pbar=show_pbar, leave_pbar=leave_pbar)
        return

    def _get_alg(self):
        self._get_SGD()
        return

    def _get_alpha(self, AHA):

        if self.lamda != 0:
            I = linop.Identity(self.x.shape)
            AHA += self.lamda * I

        max_eig = MaxEig(AHA, dtype=self.x.dtype, device=self.device,
                         max_iter=self.max_power_iter,
                         show_pbar=self.show_pbar).run()
        if max_eig == 0:
            alpha = 1
        else:
            alpha = 0.5 / max_eig

        return alpha

    def _get_SGD(self):

        if self.G is not None: #Inefficient implementation for TV

            def proxg1(alpha, x0):
                with self.device:
                    u = self.device.xp.zeros(self.G.oshape, dtype=self.y.dtype)
                    w = self.device.xp.copy(x0)

                proxfc = sp.prox.Conj(self.proxg)
                # def proxfc(sigma, v):
                #     return
                    # return v - sigma*self.proxg(1/sigma, v/sigma)

                def proxg2(tau, v):
                    v = (self.x + v/tau)/(1 + 1/tau)
                    # v = (tau * self.x + v)/(1 + tau)
                    return v

                app = sp.app.App(sp.alg.PrimalDualHybridGradient(
                    proxfc,
                    proxg2,
                    self.G,
                    self.G.H,
                    w,
                    u,
                    alpha,
                    1/alpha,
                    gamma_primal=0,
                    gamma_dual=0,
                    max_iter=self.max_cg_iter),
                    show_pbar=False)
                app.run()

                return app.alg.x

        if self.alpha is None:
            # AHA = self._get_subsampled_hessian()
            AHA = self._get_hessian_for_alpha()
            self.alpha = self._get_alpha(AHA)

        self.alg = StochasticGradient(
            self.x,
            self.alpha,
            beta=self.beta,
            device=self.device,
            proxg=self.proxg,
            max_iter=self.max_iter,
            accelerate=self.accelerate,
            tol=self.tol)
        return

    def _get_subsampled_gradient(self):
        coeff, A_S, y_S = self._get_subsampled_problem()
        def gradf(x):
            return coeff * A_S.H( A_S(x) - y_S )
        return gradf

    def _pre_update(self):
        # self.alg.sub_gradf = self._get_subsampled_gradient()
        self.alg.gradf = self._get_subsampled_gradient()
        return

    def _post_update(self):
        # if np.isinf(self.alg.resid) or np.isnan(self.alg.resid):
        #     raise OverflowError
        # return
        self.alg._update_stepsize()

        if np.isinf(self.alg.resid) or np.isnan(self.alg.resid) or \
            self.alg.resid > self.max_resid:
            raise OverflowError
        return

    def _summarize(self):
        if self.save_objective_values:
            self.objective_values.append(self.objective())

        if self.show_pbar:
            if self.save_objective_values:
                self.pbar.set_postfix(
                    obj="{0:.2E}".format(self.objective_values[-1])
                )
            else:
                self.pbar.set_postfix(
                    resid="{0:.2E}".format(
                        backend.to_device(self.alg.resid, backend.cpu_device)
                    )
                )

    def _output(self):
        return self.x

    def objective(self):
        # print(np.linalg.norm(self.x))
        with self.device:
            r = self.A(self.x) - self.y

            obj = 1 / 2 * self.device.xp.linalg.norm(r).item()**2
            if self.lamda > 0:
                if self.z is None:
                    obj += self.lamda / 2 * self.device.xp.linalg.norm(
                        self.x).item()**2
                else:
                    obj += self.lamda / 2 * self.device.xp.linalg.norm(
                        self.x - self.z).item()**2

            if self.proxg is not None:
                if self.g is None:
                    raise ValueError(
                        'Cannot compute objective when proxg is specified,'
                        'but g is not.')

                if self.G is None:
                    obj += self.g(self.x)
                else:
                    obj += self.g(self.G(self.x))

            return obj
