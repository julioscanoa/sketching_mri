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
from sigpy.alg import (PowerMethod, GradientMethod, ADMM,
                       ConjugateGradient, PrimalDualHybridGradient)

from sigpy.app import (App, MaxEig)
import scipy.io as sio
import cupy as cp
# from utils import TotalVariationRecon_PDHGmod

class SketchedLinearLeastSquares(App):
    def __init__(self, A, y, x=None,
                    proxg=None, lamda=0, G=None, g=None, z=None,
                    alpha=None, beta=1.0, tau=None, sigma=None,
                    max_outer_iter=20, max_inner_iter=5, max_power_iter=30,
                    accelerate=True, rho=1, max_cg_iter=10, tol=0,
                    save_objective_values=False,
                    device=sp.cpu_device, solver=None,
                    show_pbar=True, leave_pbar=True,
                    seed=None, max_resid=np.inf):

        self.A = A
        self.x = x
        self.y = y
        self.proxg = proxg
        self.lamda = lamda
        self.G = G
        self.g = g
        self.solver = solver

        self.alpha = alpha
        self.sigma = sigma
        self.max_resid = max_resid

        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.max_iter = max_outer_iter * max_inner_iter
        self.max_power_iter = max_power_iter
        self.max_cg_iter = max_cg_iter
        self.accelerate = accelerate
        self.outer_iter = 0

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

        self.Ahy = None
        self._get_Ahy()
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
        self._get_GradientMethod()

        return

    def _get_alpha(self):

        self.alpha = np.zeros(self.max_outer_iter, dtype=np.float32)
        num_est = 4
        beta = .9
        for isk in range(min(self.max_outer_iter, num_est)):
            A_S = self._get_sketched_problem(isk)
            AHA = A_S.N

            if self.lamda != 0:
                I = linop.Identity(self.x.shape)
                AHA += self.lamda * I

            max_eig = MaxEig(AHA, dtype=self.x.dtype, device=self.device,
                            max_iter=self.max_power_iter,
                            show_pbar=self.show_pbar).run()
            if max_eig == 0:
                alpha = 1
            else:
                alpha = 1 / max_eig
            alpha *= beta**(isk//2)
            self.alpha[isk] = alpha

        self.alpha[num_est:] = min(self.alpha[:num_est])
        return 

    def _get_GradientMethod(self):

        if self.alpha is None:
            self._get_alpha()

        self.alg = GradientMethod(
            None,
            self.x,
            self.alpha[0],
            proxg=self.proxg,
            max_iter=self.max_iter,
            accelerate=self.accelerate,
            tol=self.tol,
        )
        return
    
    def _get_true_gradient(self):
        with self.device:
            self.d = self.A.H(self.A(self.x)) - self.Ahy
        return self.d
    
    def _get_Ahy(self):
        with self.device:
            self.Ahy = self.A.H(self.y)
        return
    
    def _get_sketched_problem(self, ind_sketch):
        raise NotImplementedError

    def _get_sketched_gradient(self):
        self.A_S = self._get_sketched_problem(self.outer_iter)
        d = self._get_true_gradient()

        with self.device:
            x0 = self.alg.x.copy()

        def gradf(x):
            gradf_x = self.A_S.N(x - x0) + d
            return gradf_x
        return gradf

    def _pre_update(self):

        if self.alg.iter % self.max_inner_iter == 0:
            self.alg.t = 1.0
            self.alg.z = self.alg.x.copy()
            self.alg.gradf = self._get_sketched_gradient()
            self.alg.alpha = self.alpha[self.outer_iter]
            self.outer_iter += 1
        return

    def _post_update(self):

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


