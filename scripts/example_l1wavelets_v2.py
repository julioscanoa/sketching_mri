import os
import sys
sys.path.append('../src')
sys.path.append('../lib')
import numpy as np
import matplotlib.pyplot as plt
import sigpy      as sp
import sketching_mri_app_v2 as sk
import stochastic_mri_app as st
from utils import coil_compression
import argparse

def main(args):
    # Setup
    devnum = args.device # Device to use, set to -1 if running in CPU
    nc = 8 # number of coils after coil compression
    nch = 3 # reduced number of coils
    R = 2.5
    img_shape = [256,256]
    lamda = 5e-2

    #Device configuration
    device = sp.Device(devnum)
    xp     = device.xp
    device.use()
    mvd = lambda x: sp.to_device(x, devnum)
    mvc = lambda x: sp.to_device(x, sp.cpu_device)

    #Loading data
    npzfile = np.load('../data/sample_radial.npz')
    ksp = npzfile['ksp']
    coord = npzfile['coord']
    dens = npzfile['dens']

    [nc0, nviews0, nread] = ksp.shape
    nviews = int(nviews0/R)

    # Coil compression steps
    ksp, _ = coil_compression(ksp)

    # Loading into device
    ksp = mvd(ksp[:nc,...])
    coord = mvd(coord)
    dens = mvd(dens)

    ksp_us = mvd(ksp[:nc,:nviews,...])
    coord_us = mvd(coord[:nviews,...])
    dens_us = mvd(dens[:,:nviews,...])

    #==========================    SMaps & US image ====================================
    if args.verbose:
        print("Sensitivity map estimation")
    JsenseApp = sp.mri.app.JsenseRecon(ksp_us, coord=coord_us, mps_ker_width=6, lamda=1e-1,
                         img_shape=img_shape, device=device, max_iter=30, show_pbar=args.verbose)
    mps = JsenseApp.run()
    mps_ker = JsenseApp.mps_ker

    if args.verbose:
        print("Done!")

    if args.verbose:
        print("Fully-sampled reference")
    # Fully-sampled reference
    img_fs = sp.mri.app.SenseRecon(ksp, mps, coord=coord, weights=dens, lamda=1e-3,
                                            device=device, max_iter=10, show_pbar=args.verbose).run()
    img_fs = mvc(img_fs)
    if args.verbose:
        print("Done!")

    # Undersampled image
    A_us = sp.mri.linop.Sense(mps, coord=coord_us, weights=dens_us**2)
    img_us = np.rot90(mvc(A_us.H*ksp_us), k=-1)
    scale = np.percentile(np.abs(img_us.flatten()), 95)

    #==========================    Reconstructions  ====================================
    if args.verbose:
        print("Starting reconstructions")

    # Baseline reconstruction with C=8
    max_iter = 20
    img_base = scale*sp.mri.app.L1WaveletRecon(ksp_us/scale, mps, lamda, weights=dens_us,
                            coil_batch_size = nch,
                            coord=coord_us, max_iter=max_iter, device=device,
                            show_pbar=args.verbose).run()
    img_base = np.rot90(mvc(img_base), k=-1)

    # Aggressive coil compression reconstruction with C=3
    max_iter = 20
    img_cc = scale*sp.mri.app.L1WaveletRecon(ksp_us[:nch,...]/scale, mps[:nch,...], lamda,
                             weights=dens_us, coord=coord_us, max_iter=max_iter, device=device,
                            show_pbar=args.verbose).run()
    img_cc = np.rot90(mvc(img_cc), k=-1)

    # AccProxSGD reconstruction with C=3
    max_iter = 20
    np.random.seed(1)
    img_sgd = scale*st.StochasticL1WaveletRecon(ksp_us/scale, mps, lamda, nch,
                                     weights=dens_us, coord=coord_us, device=device,
                                     max_iter=max_iter, beta=0.95,
                                     show_pbar=args.verbose).run()
    img_sgd = np.rot90(mvc(img_sgd), k=-1)

    # Coil sketching reconstruction with C=3
    max_inner_iter = 5
    max_outer_iter = 4

    mps_S = sk.SketchCoils(mps_ker, img_shape, nch, max_outer_iter, seed=1).run()

    img_sk = scale*sk.SketchedL1WaveletRecon(ksp_us/scale, mps, lamda, nch,
                                    mps_S=mps_S,
                                    weights=dens_us, coord=coord_us,
                                    show_pbar=args.verbose, device=device,
                                    max_outer_iter=max_outer_iter,
                                    max_inner_iter=max_inner_iter).run()
    img_sk = np.rot90(mvc(img_sk), k=-1)

    if args.verbose:
        print("Done!")
    #==========================    Show images ====================================
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.save_plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,8))
        fig.tight_layout()
        ax1.imshow(np.abs(img_base), cmap='gray', vmin=0, vmax=.6)
        ax1.axis('off')
        ax1.set_title('Baseline $C=8$')

        ax2.imshow(np.abs(img_cc), cmap='gray', vmin=0, vmax=.6)
        ax2.axis('off')
        ax2.set_title('Aggressive Coil compression $\hat{C}=3$')

        ax3.imshow(np.abs(img_sgd), cmap='gray', vmin=0, vmax=.6)
        ax3.axis('off')
        ax3.set_title('AccProxSGD $\hat{C}=3$')

        ax4.imshow(np.abs(img_sk), cmap='gray', vmin=0, vmax=.6)
        ax4.axis('off')
        ax4.set_title('Coil sketching $\hat{C}=3$')

        file_plot = os.path.join(args.output_dir + 'example_l1wav.png')
        plt.savefig(file_plot)
        if args.verbose:
            print("Plot saved in {:s}".format(args.output_dir))

    if args.save_npz:
        file_npz = os.path.join(args.output_dir + 'example_l1wav.npz')
        np.savez(file_npz, img_base=img_base, \
                    img_cc=img_cc, img_sgd=img_sgd, img_sk=img_sk)

        if args.verbose:
            print("Npz file saved in {:s}".format(args.output_dir))


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
    parser.add_argument('--output_dir', type=str, default='../results/', help='Directory to output files')
    parser.add_argument('--save_plot', default=True, action='store_true', help='Turn on save plot')
    parser.add_argument('--save_npz', action='store_true', help='Turn on save npz with the images')
    parser.add_argument('--device', type=int, default=-1, help='GPU device')
    parser.add_argument('--verbose', action='store_true', help='Turn on debug statements')
    return parser


if __name__ == '__main__':
    print('================= Coil sketching ==========================')
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
    print('Script complete.')
