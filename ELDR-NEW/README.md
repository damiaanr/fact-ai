# General instructions

All code is run through `main.py` in command-line, without command line arguments/parameters. Note that all VAE models have been pre-trained (by us), and that all explanations for all datasets and dimensionality reduction techniques, for all levels of sparsitity, have already been learnt and provided. If you wish to re-train and re-learn, please delete all files inside the `results/deltas`, `results/vae_models` folders. Optionally, you could also delete all plot-data from the `results/measures` and `results/plots` folders.

By simply toggling straightfoward variables in the code of `main.py`, you can:

  1. Learn explanations (which might involve training VAE models, if not yet trained): set `train_algos` on line `137` to `True`. This will generate 'latent space plots' and 'performance plots' which are also stored in the `results/plots` folder.
      - You can optionally skip the graphs of latent spaces and metrics (for example, when bulk-training all datasets) by setting `skip_graphs` on line `140` to `True`
      - You can chose to load pre-saved deltas, instead of re-learning these (VERY much recommended!), by setting `load_saved_deltas` on line `139` to `True`
  2. Plot explanations: set `show_explanation` on line `83` to `True` and edit the corresponding data on line `84`. This will also generate a 'metrics plot'. All plots will be stored in the `results/plots` folder.
      - You can optionally choose to only plot the 'metrics plot' without restricting to a specific K, for this, set K to `None` on line `84`.
  3. Re-plot a 'performance' plot generated in step 1. The data for these plots are stored in the `results/measures` folder. Set `show_plot` on line `41` to `True`, and specify the file name on line `44`.
  
Optionally, you could also:

  - Set the minimum variance (for PCA, SPCA, KPCA, ISO, LLE) on line `61`
  - Change the number of training trials (per lambda, per K, per algorithm, per dataset) on line `5`
  - Change the different to-attempt regularization parameters (not recommended) on line `197`
  - Change the verbose interval of the explanation-learning method on line `200`
  - Set an additional scale factor for datasets on line `93` (currently only for the Glass dataset)
  
Enjoy!

# Different dimensionality reduction algorithms

All dimensionality reduction algorithms are stored in `dimensionality_reduction_algorithms.py`. It is possible, but not recommended, to disable some of these dimensionality reduction algorithms by simpling 'commenting out' those of choice on lines `27-33`. Please note that the all dimensionality reduction algorithms, except VAE, are loaded through external libraries. The code for VAE is inside the `VAE` folder.

# Different datasets

All data is stored in the `data` folder. New lines separate samples, while tabs separate the features of a sample. As we are using K-means for clustering, we do not need more than X. It is possible, and also recommended, to only load a few of the datasets. The datasets which do not have to be loaded can be simply 'commented out' on lines `65-70` of `main.py`.

# Packages to install
This code was run on Python 3.7 with PyTorch 1.7.0.
To quickly install the required package:
```
pip3 install numpy matplotlib torch sklearn pandas scipy
```