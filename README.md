# Coil sketching
Code to reproduce results in manuscript "Coil Sketching for computationallly-efficient MR iterative reconstruction".

Written by Julio A. Oscanoa (joscanoa@stanford.edu), 2023.

## Installation
To run the code in this package, we recommend conda. We recommend downloading and installing [Miniconda3](https://docs.conda.io/en/latest/miniconda.html).

After installing conda, create your conda environment from `environment.yaml` using the following command:
```bash
conda env create -f environment.yaml
```

Then, activate the environment:
```bash
conda activate sketching-env
```

## Reconstruction
In the scripts folder, we provide a python script and a jupyter notebook with an example reconstruction with radial data.

### Python script
The script by default runs in CPU. By default, the script will run the reconstructions and store a plot of the images.
Run the following:

```bash
cd scripts
python example_l1wavelets.py
```

Alternatively, to use GPU and enable verbose, run the following:
```bash
cd scripts
python example_l1wavelets.py --device=0 --verbose
```

### Jupyter notebook
Install the jupyter lab packages and enable the ipywidgets notebook extension for showing progress bars.

```bash
conda install -c conda-forge notebook=6.3
conda install -c conda-forge jupyterlab=3.2
conda install -c conda-forge ipywidgets=7.6
jupyter nbextension enable --py widgetsnbextension
```

### Version 2.0 notes
Version 2.0 of the coil sketching code is uploaded. Reconstruction classes are lighter and more efficient.
Currently, this new version includes only the L1-Wavelets reconstruction. This will be updated in the future.
Python script and jupyter notebooks for this version are included as well.