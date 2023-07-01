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
from sigpy.alg import (PowerMethod, GradientMethod, ADMM,
                       ConjugateGradient, PrimalDualHybridGradient)
from sigpy.app import LinearLeastSquares
# from utils import TotalVariationRecon_PDHGmod

class SketchedLinearLeastSquares(LinearLeastSquares):
    def __init__(self, A, y, max_init_iter=30, max_outer_iter=20, max_inner_iter=5,
                 alpha_init=None, sigma_init=None, tau_init=None, num_alphas=None,
                 device=sp.cpu_device, seeds=None, **kwargs):

        self.A_S = None
        self.y_S= None
        self.sigma_t = None
        self.d = None #true gradient
        self.AHy = None

        self.alpha_init = alpha_init
        self.sigma_init = sigma_init
        self.tau_init = tau_init
        self.num_alphas = num_alphas

        self.max_init_iter = max_init_iter
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter - 1
        self.iter = 0
        self.outer_iter = 0
        kwargs['max_iter'] = self.max_init_iter + self.max_outer_iter * self.max_inner_iter
        self.device = device

        if self.num_alphas is None:
            self.num_alphas = self.max_outer_iter // 2

        self.y = y
        self._get_AHy()
        if self.max_outer_iter > 0:
            self._make_sketched_models_A_S()
        super().__init__(A, y, **kwargs)

        with self.device:
            self.x0 = self.device.xp.copy(self.x)
        if self.save_objective_values:
            self.y = sp.to_device(self.y, device=self.device)
        else:
            self.y = None
            self.A = None

        self.nseed = 0

    def _get_alg(self):

        if self.solver is None:
            if self.proxg is None:
                self.solver = 'ConjugateGradient'

            elif self.G is None:
                self.solver = 'GradientMethod'

            else:
                self.solver = 'PrimalDualHybridGradient'

        if self.solver == 'GradientMethod':

            # Alphas
            if self.alpha is None and self.max_outer_iter > 0:

                self.alpha = np.ones((self.max_outer_iter,)) * np.inf

                for i in range(self.num_alphas):
                    # self._make_sketched_model_A()
                    self._load_sketched_model_A_S()
                    self.outer_iter +=1

                    AHA = self.A_S.H * self.A_S

                    if self.lamda != 0:
                        I = linop.Identity(self.x.shape)
                        AHA += self.lamda * I

                    max_eig = sp.app.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                                     max_iter=self.max_power_iter,show_pbar=self.show_pbar).run()

                    if max_eig == 0:
                        alpha = 1
                    else:
                        alpha = 1 / max_eig

                    self.alpha[i] = alpha

                self.outer_iter = 0
                #Complete alphas with the min alpha
                if self.num_alphas < self.max_outer_iter:
                    min_alpha = min(self.alpha)
                    self.alpha[self.num_alphas:] = min_alpha

                if self.max_init_iter == 0:
                    self._update_GradientMethod()

            # Alpha for initialization
            if self.alpha_init is None and self.max_init_iter > 0:
                self._make_initial_sketched_problem()
                AHA = self.A_S.H * self.A_S

                if self.lamda != 0:
                    I = linop.Identity(self.x.shape)
                    AHA += self.lamda * I
                max_eig = sp.app.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                                 max_iter=self.max_power_iter,show_pbar=self.show_pbar).run()

                if max_eig == 0:
                    self.alpha_init = 1
                else:
                    self.alpha_init = 1 / max_eig

                self._get_init_GradientMethod()

        elif self.solver == 'PrimalDualHybridGradient':

            # initial step size
            if  self.max_init_iter > 0:
                if self.sigma_init is None:
                    if self.tau_init is None:
                        self._make_initial_sketched_problem()
                        AHA = self.A_S.H * self.A_S
                        max_eig = sp.app.MaxEig(
                            AHA,
                            dtype=self.y.dtype,
                            device=self.device,
                            max_iter=self.max_power_iter,
                            show_pbar=self.show_pbar).run()

                        self.tau_init = 1 / max_eig

                    G = self.G
                    S = sp.linop.Multiply(G.oshape, self.tau_init)
                    GHG = G.H * S * G

                    max_eig = sp.app.MaxEig(
                        GHG,
                        dtype=self.y.dtype,
                        device=self.device,
                        max_iter=self.max_power_iter,
                        show_pbar=self.show_pbar).run()

                    self.sigma_init = 1 / max_eig

                elif self.tau_init is None:
                    self._make_initial_sketched_problem()
                    S = sp.linop.Multiply(self.A_S.oshape, self.sigma_init)
                    AHA = self.A_S.H * S * self.A_S

                    max_eig = sp.app.MaxEig(
                        AHA,
                        dtype=self.y.dtype,
                        device=self.device,
                        max_iter=self.max_power_iter,
                        show_pbar=self.show_pbar).run()

                    self.tau_init = 1 / max_eig

                self._get_init_PrimalDualHybridGradient()

            # Step sizes
            if self.max_outer_iter > 0:
                if self.sigma is None:
                    if self.tau is None:
                        self._make_sketched_model_A()
                        AHA = self.A_S.H * self.A_S
                        max_eig = sp.app.MaxEig(
                            AHA,
                            dtype=self.y.dtype,
                            device=self.device,
                            max_iter=self.max_power_iter,
                            show_pbar=self.show_pbar).run()
                        tau = 0.2 / max_eig
                        # tau = 1
                        self.tau = tau

                    G = self.G
                    S = sp.linop.Multiply(G.oshape, tau)
                    GHG = G.H * S * G

                    max_eig = sp.app.MaxEig(
                        GHG,
                        dtype=self.y.dtype,
                        device=self.device,
                        max_iter=self.max_power_iter,
                        show_pbar=self.show_pbar).run()
                    sigma = 0.5 / max_eig
                    # sigma = 1
                    self.sigmas = sigma

                elif self.tau is None:
                    self._make_sketched_model_A()
                    S = sp.linop.Multiply(self.A_S.oshape, self.sigma)
                    AHA = self.A_S.H * S * self.A_S

                    max_eig = sp.app.MaxEig(
                        AHA,
                        dtype=self.y.dtype,
                        device=self.device,
                        max_iter=self.max_power_iter,
                        show_pbar=self.show_pbar).run()

                    tau = 1 / max_eig
                    self.tau = tau

                if self.alg is None:
                    self._get_PrimalDualHybridGradient()
        return

    def _get_true_gradient(self):
        with self.device:
            self.d = self.A.H(self.A(self.x) - self.y)
        return

    def _make_sketched_models_A_S(self):
        raise NotImplementedError

    def _make_initial_sketched_problem(self):
        raise NotImplementedError

    def _get_init_GradientMethod(self):
        with self.device:
            AHy = self.A_S.H(self.y_S)
        def gradf(x):
            with self.device:
                gradf_x = self.A_S.N(x) - AHy
                if self.lamda != 0:
                    if self.z is None:
                        util.axpy(gradf_x, self.lamda, x)
                    else:
                        util.axpy(gradf_x, self.lamda, x - self.z)

                return gradf_x

        self.alg = GradientMethod(
            gradf,
            self.x,
            self.alpha_init,
            proxg=self.proxg,
            max_iter=self.max_iter,
            accelerate=self.accelerate,
            tol=self.tol,
        )

    def _get_init_ConjugateGradient(self):
        AHA = self.A_S.N

        if self.lamda != 0:
            AHA += self.lamda * linop.Identity(self.x.shape)
            if self.z is not None:
                util.axpy(AHy, self.lamda, self.z)

        self.alg = ConjugateGradient(
            AHA, self.AHy, self.x, P=self.P, max_iter=self.max_iter, tol=self.tol
        )

    def _get_init_PrimalDualHybridGradient(self):

        gamma_primal = 0
        gamma_dual = 0

        with self.y_device:
            if self.G is None:
                self.G = sp.linop.Identity(self.x0.shape)

            H = self.A_S.H*self.A_S
            I = sp.linop.Identity(self.x.shape)
            b = self.AHy

            def proxfc(sigma, v):
                return v - sigma*self.proxg(1/sigma, v/sigma)

            def proxg(tau, v):
                sp.app.App(sp.alg.ConjugateGradient(H + (1/tau)*I, b + v/tau, v,
                            max_iter=self.max_cg_iter), show_pbar=False).run()
                return v

        with self.y_device:
            u = self.y_device.xp.zeros(self.G.oshape, dtype=self.x.dtype)

        self.alg = sp.alg.PrimalDualHybridGradient(
            proxfc,
            proxg,
            self.G,
            self.G.H,
            self.x,
            u,
            self.tau_init,
            self.sigma_init,
            gamma_primal=gamma_primal,
            gamma_dual=gamma_dual,
            max_iter=self.max_iter,
            tol=self.tol)

    def _update_alg(self):
        # Check if algorithms variabes are corrrectly set!!
        if self.solver == 'GradientMethod':
            if self.G is not None:
                raise ValueError('GradientMethod cannot have G specified.')
            self._get_GradientMethod()
        elif self.solver == 'PrimalDualHybridGradient':
            self._get_PrimalDualHybridGradient()
        # elif self.solver == 'ADMM':
        #     self._get_ADMM()
        elif self.solver == 'ConjugateGradient':
            self._get_ConjugateGradient()
        else:
            raise ValueError('Invalid solver: {solver}.'.format(
                solver=self.solver))
        return

    def _get_ConjugateGradient(self):
        I = linop.Identity(self.x.shape)
        AHA = self.A_S.N
        AHy = self.A_S.N(self.x0) - self.d
        # AHy = -self.d

        if self.lamda != 0:
            AHA += self.lamda * I
            if self.z is not None:
                util.axpy(AHy, self.lamda, self.z)

        if self.alg is not None:
            self.iter = self.alg.iter

        self.alg = ConjugateGradient(
            AHA, AHy, self.x, P=self.P, max_iter=self.max_iter)
        self.alg.iter = self.iter

        return

    def _get_GradientMethod(self):

        # First iteration
        self.x -= np.mean(self.alpha) * self.d
        if self.proxg is not None:
            self.x = self.proxg(np.mean(self.alpha), self.x)

        def gradf(x):
            with self.device:
                gradf_x = self.A_S.N(x - self.x0) + self.d

                if self.lamda != 0:
                    if self.z is None:
                        util.axpy(gradf_x, self.lamda, x)
                    else:
                        util.axpy(gradf_x, self.lamda, x - self.z)

                return gradf_x

        if self.alg is not None:
            self.iter = self.alg.iter

        self.alg = GradientMethod(
                            gradf,
                            self.x,
                            np.min(self.alpha),
                            proxg=self.proxg,
                            max_iter=self.max_iter,
                            accelerate=self.accelerate)
        self.alg.iter = self.iter
        return

    def _get_PrimalDualHybridGradient(self):

        with self.device:
            if self.G is None:
                self.G = sp.linop.Identity(self.x0.shape)

            H_S = self.A_S.H*self.A_S
            I = sp.linop.Identity(self.x0.shape)
            b = self.A_S.H(self.A_S(self.x0)) - self.d

            proxFc = sp.prox.Conj(self.proxg)
            # def proxFc(sigma, v):
            #     return v - sigma*self.proxg(1/sigma, v/sigma)

            def proxG(tau, v):
                sp.app.App(sp.alg.ConjugateGradient(H_S + (1/tau)*I, b + v/tau, v,
                            max_iter=self.max_cg_iter), show_pbar=False).run()
                return v

        if self.alg is not None:
            self.iter = self.alg.iter

        gamma_primal = 0
        gamma_dual = 0
        with self.device:
            u = self.device.xp.zeros(self.G.oshape, dtype=self.y.dtype)

        self.alg = PrimalDualHybridGradient(
            proxFc,
            proxG,
            self.G,
            self.G.H,
            self.x,
            u,
            self.tau,
            self.sigma,
            gamma_primal=gamma_primal,
            gamma_dual=gamma_dual,
            max_iter=self.max_iter)
        self.alg.iter = self.iter + 1

        return

    def _pre_update(self):

        if self.alg.iter >= self.max_init_iter:
            if (self.alg.iter - self.max_init_iter) % self.max_inner_iter == 0 :
                backend.copyto(self.x0, self.x)
                self._get_true_gradient()
                # self._make_sketched_model_A()
                self._load_sketched_model_A_S()
                self._update_alg()
                self.outer_iter += 1
        return
