from explainer import Explainer
from functions import load_data, plot_algorithm_comparison, cluster_latent_space
from dimensionality_reduction_algorithms import generate_transformers
import numpy as np
import pickle
import math
import sys
import os.path
from datetime import datetime

import matplotlib.pyplot as plt

global_dir = "."
sys.path.append(global_dir)


# Note to reader/teacher/instructor:

# This file, main.py, is the `gateway` to all the magic we created.
# By toggling variables (such as show_plot, show_explanation, train_algos),
# according to the comments stated in the document, it is possible to
# execute anything from training models, to learning explanations, to plotting
# different kinds of performance plots. This file should be run in commandline
# without any arguments. We'll like to keep it simple for now.

# Enjoy!









############################################
## THE CODE BELOW IS USED FOR PLOTTING DATA#
## OF ALREADY PERFORMED EXPERIMENTS ########
############################################

show_plot = False

if show_plot:
  with open(global_dir + '/results/measures/Housing_1611937457.305099.pickle', 'rb') as handle:
    results = pickle.load(handle)
    plot_algorithm_comparison(results)
    quit()
    
    
    

    
    
############################################
## DATASETS TO BE LOADED CAN BE CHANGED    #
## ACCORDING TO WISHES BELOW ###############
############################################
    
print('Loading datasets...')

variance_adjust = 10 # The variance we would like the data in our latent spaces to have

# Loading all datasets - tuples: (X, num_clusters)
datasets = {
  'Housing': (load_data(global_dir + '/data/Housing.tsv', False, False, False)[0], 6),
  'Iris': (load_data(global_dir + '/data/Iris.tsv', False, False, False)[0], 3),
  'Heart': (load_data(global_dir + '/data/Heart.tsv', False, False, False)[0], 8),
  'Seeds': (load_data(global_dir + '/data/Seeds.tsv', False, True, True)[0], 3),
  'Wine': (load_data(global_dir + '/data/Wine.tsv', False, True, False)[0], 3),
  'Glass': (load_data(global_dir + '/data/Glass.tsv', False, True, True)[0], 7),
}






############################################
## THE CODE BELOW IS USED FOR PLOTTING DATA#
## OF EXPLANATIONS/MEASURES ################
############################################

load_saved_latent_data = True
show_explanation = False
k, c1, c2, dataset, algo = (13, 1, 4, 'Housing', 'vae') # for ex: (7, 1, 4, 'Housing', 'vae')
                                                       # note: if K is 'None', global metrics will be plotted

if show_explanation:
  if dataset not in datasets.keys():
    print('Dataset was not loaded - can not plot explanation')
  else:
    original_X, num_clusters = datasets[dataset]
    
    transform_functions = generate_transformers(original_X, dataset, global_dir, min_variance = variance_adjust, additional_scale_tsvd = (20 if dataset == 'Glass' else 1))
    
    if algo not in transform_functions.keys():
      print('Algorithm was not loaded - can not plot explanaton')
    else:
      transformer = transform_functions[algo]
      
      latent_x_file = global_dir + "/results/latent_data/" + dataset + "_" + algo + ".pickle"
      
      if not load_saved_latent_data or not os.path.isfile(latent_x_file):
        latent_X    = transformer(original_X)
        latent_X[~np.isfinite(latent_X)] = 0
        latent_Y, latent_Y_centers = cluster_latent_space(latent_X, num_clusters)
        
        latent_representation = (latent_X, latent_Y, latent_Y_centers)
        
        with open(latent_x_file, 'wb') as handle:
          pickle.dump(latent_representation, handle, protocol=pickle.HIGHEST_PROTOCOL)
      else:
        with open(latent_x_file, 'rb') as handle:
          latent_X, latent_Y, latent_Y_centers = pickle.load(handle)
          print("Loading pre-saved latent data for dataset %s on %s" % (dataset, algo))
      
      if k is None:
        kf = original_X.shape[1] if original_X.shape[1] <= 5 else (original_X.shape[1] if original_X.shape[1] % 2 == 1 else original_X.shape[1]-1)
      else:
        kf = k
      
      save_file_name = global_dir + "/results/deltas/" + dataset + "_" + algo + "_" + str(variance_adjust) + "_k" + str(kf) + ".npy"
        
      if not os.path.isfile(save_file_name):
        print('The explanation for this settings has not been learnt yet!')
      else:
        E = Explainer(original_X, latent_Y, transformer, num_clusters, 0.5, global_dir, latent_X)
        E.set_delta(np.load(save_file_name))
        cr, cv = E.metrics(k=k)
        
        print('Average correctness: %.3f | Average coverage: %.3f' % (np.mean(cr), np.mean(cv)))
        
        # First, we plot the metrics for this K
        E.plot_metrics()
        
        if k is not None:
          # Now, we plot the explanation
          E.plot_explanation(c1, c2, k)
        
  quit()






############################################
## THE CODE BELOW IS USED FOR EXPERIMENTS ##
############################################

train_algos       = True
num_trials        = 5
load_saved_deltas = True # Training will be skipped if saved data is available
skip_graphs       = False # Used when idle/afk training datasets after each other

# Preload 

print('Finished loading %d dataset(s)' % len(datasets))

for dataset in datasets.keys():
  original_X, num_clusters = datasets[dataset]

  # Generate transformation functions (r - map from original to latent space)
  transform_functions = generate_transformers(original_X, dataset, global_dir, min_variance = variance_adjust, additional_scale_tsvd = (20 if dataset == 'Glass' else 1))
  
  # We will be training for different k with different algorithms
  K = np.arange(1, original_X.shape[1]+1, (1 if original_X.shape[1] <= 5 else 2))
  out = np.zeros((len(K), 2*len(transform_functions))) # correctness and coverage
    
  fig, axs = plt.subplots((3 if len(transform_functions) >= 3 else len(transform_functions)), math.ceil(len(transform_functions)/3))
    
  i = 0
  for dr_algorithm in transform_functions.keys():
    print('==> Applying dimensionality reduction using %s on the %s dataset' % (dr_algorithm, dataset))
    transformer = transform_functions[dr_algorithm]
    
    # Loading existing latent spaces (or creating and dumping)
    latent_x_file = global_dir + "/results/latent_data/" + dataset + "_" + dr_algorithm + ".pickle"
      
    if not load_saved_latent_data or not os.path.isfile(latent_x_file):
      latent_X = transformer(original_X)
      latent_X[~np.isfinite(latent_X)] = 0
      latent_Y, latent_Y_centers = cluster_latent_space(latent_X, num_clusters)
      
      latent_representation = (latent_X, latent_Y, latent_Y_centers)
      
      with open(latent_x_file, 'wb') as handle:
        pickle.dump(latent_representation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      with open(latent_x_file, 'rb') as handle:
        latent_X, latent_Y, latent_Y_centers = pickle.load(handle)
        print("Loading pre-saved latent data for dataset %s on %s" % (dataset, dr_algorithm))
    
    
    # Plotting latent space
    if not skip_graphs:
      axs[int(i/2 % 3), math.floor(i/2/3)].scatter(latent_X[:, 0], latent_X[:, 1], c=latent_Y, s=2, cmap='viridis')
      axs[int(i/2 % 3), math.floor(i/2/3)].scatter(latent_Y_centers[:, 0], latent_Y_centers[:, 1], c='black', s=200, alpha=0.5);
      axs[int(i/2 % 3), math.floor(i/2/3)].set_title(dr_algorithm)
    
    # Training and evaluating for different Ks and lambdas
    if train_algos:
      j = 0
      
      for k in K: # We run 5 trials for 11 lambda's for every K (for every algo for every dataset)
        best_measure = 0.0
        save_file_name = global_dir + "/results/deltas/" + dataset + "_" + dr_algorithm + "_" + str(variance_adjust) + "_k" + str(k) + ".npy"
        
        if load_saved_deltas and os.path.isfile(save_file_name):
          print('Loading pre-saved delta for %s, K = %d' % (dr_algorithm.upper(), k))
          
          E = Explainer(original_X, latent_Y, transformer, num_clusters, 0.5, global_dir, latent_X)
          E.set_delta(np.load(save_file_name))
          
          cr, cv = E.metrics(k=k)
          
          mean_cr = np.mean(cr)
          mean_cv = np.mean(cv)
          
          out[j, i]   = mean_cr
          out[j, i+1] = mean_cv
        else:
          for lamb in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            for trial in range(num_trials):
              E = Explainer(original_X, latent_Y, transformer, num_clusters, lamb, global_dir, latent_X)
              E.learn(verbose_interval=1000)
              cr, cv = E.metrics(epsilon=1.0, k=k)
              
              mean_cr = np.mean(cr)
              mean_cv = np.mean(cv)
              
              print('[%s, K = %d, λ = %.1f, trial %d] Average correctness: %.3f | Average coverage: %.3f' % (dr_algorithm.upper(), k, lamb, trial+1, mean_cr, mean_cv))
              
              if mean_cr > best_measure:
                best_measure = mean_cr
                out[j, i]   = mean_cr
                out[j, i+1] = mean_cv
                print('New record! Saved delta to ' + str(save_file_name))
                np.save(save_file_name, E.get_delta())
        j += 1
    i += 2 # index for DR algorithm in output matrix
    
  if not skip_graphs:
    # First showing the latent space plot
    if len(transform_functions) % 2 == 1:
      axs[-1, -1].axis('off')
      if len(transform_functions) == 7:
        axs[-2, -1].axis('off')
        
    fig.suptitle(dataset + " dataset (variance adjusted to " + str(variance_adjust) + ", " + str(num_clusters) + " clusters)")
    plt.show()
    fig.savefig(global_dir + "/results/plots/latent_spaces_" + dataset + "_" + str(datetime.timestamp(datetime.now())) + ".png", bbox_inches='tight')
    
    # Then plotting corresponding results (if training was enabled)
    if train_algos:
      results = (out, K, list(transform_functions.keys()), dataset)
      
      plot_algorithm_comparison(results)
      
      with open(global_dir + "/results/measures/" + dataset + "_" + str(datetime.timestamp(datetime.now())) + ".pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  