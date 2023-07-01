import numpy as np
import sigpy as sp
import sys
import os
import time
from tqdm.auto import tqdm
import scipy.io as sio
sys.path.append('../src')
sys.path.append('../lib')
import cupy as cp

def coil_compression(ksp):
    device = sp.get_device(ksp)
    ksp_shape = ksp.shape
    nc = ksp_shape[0]

    E = device.xp.reshape(ksp, [nc, -1])
    E = device.xp.swapaxes(E, 0, 1)

    EHE = device.xp.matmul( E.conj().T, E)
    eigval, eigvec = device.xp.linalg.eigh(EHE)

    eigval = device.xp.flip(eigval)
    eigvec = device.xp.flip(eigvec, axis=1)

    E = device.xp.matmul(E, eigvec)
    ksp = device.xp.reshape(device.xp.swapaxes(E, 0, 1), ksp_shape)

    return ksp, eigval

def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Mean Squared Error """

    error = np.abs(actual - predicted)
    rmse = np.sqrt(np.mean(np.square(error), axis=(-1,-2)))
    range = np.ptp(np.abs(actual), axis=(-1,-2))

    return rmse/range
