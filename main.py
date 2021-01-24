from explainer import Explainer
from functions import load_data, plot_algorithm_comparison
from dimensionality_reduction_algorithms import generate_transformers
import numpy as np
import pickle
import math
import sys
import os.path
from datetime import datetime

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

global_dir = "."

sys.path.append(global_dir)

############################################
## THE CODE BELOW IS USED FOR PLOTTING DATA#
## OF ALREADY PERFORMED EXPERIMENTS ########
############################################

show_plot = False

if show_plot:
  with open(global_dir + '/results/measures/Iris_1611441106.107953.pickle', 'rb') as handle:
    results = pickle.load(handle)
    plot_algorithm_comlsparison(results)
    quit()

############################################
## THE CODE BELOW IS USED FOR EXPERIMENTS ##
############################################

train_algos       = True
variance_adjust   = 10
num_trials        = 5
load_saved_deltas = True # Training will be skipped if saved data is available

run_time          = str(datetime.timestamp(datetime.now()))

print('Loading datasets...')

# Loading all datasets - tuples: (X, num_clusters)
datasets = {
  #'Bipolar': (load_data(global_dir + '/data/Bipolar.tsv', False, False, False)[0], 18),
  'Housing': (load_data(global_dir + '/data/Housing.tsv', False, False, False)[0], 6),
  #'Iris': (load_data(global_dir + '/data/Iris.tsv', False, False, False)[0], 3),
  # 'Heart': (load_data(global_dir + '/data/Heart.tsv', False, False, False)[0], 8),
  #'Seeds': (load_data(global_dir + '/data/Seeds.tsv', False, True, True)[0], 3),
  #'HTRU': (load_data(global_dir + '/data/HTRU.tsv', False, False, True)[0], 2),
  #'Wine': (load_data(global_dir + '/data/Wine.tsv', False, True, False)[0], 3),
  #'Ecoli': (load_data(global_dir + '/data/Ecoli.tsv', False, True, True)[0], 8),
  #'Glass': (load_data(global_dir + '/data/Glass.tsv', False, True, True)[0], 7),
  #'Accents': (load_data(global_dir + '/data/Accents.tsv', False, True, False)[0], 6),
}

# Preload 

print('Finished loading %d datasets' % len(datasets))

for dataset in datasets.keys():
  original_X, num_clusters = datasets[dataset]

  # Generate transformation functions (r - map from original to latent space)
  transform_functions = generate_transformers(original_X, min_variance = variance_adjust)
  
  # We will be training for different k with different algorithms
  K = np.arange(1, original_X.shape[1]+1, (1 if original_X.shape[1] <= 5 else 2))
  out = np.zeros((len(K), 2*len(transform_functions))) # correctness and coverage
    
  fig, axs = plt.subplots((3 if len(transform_functions) >= 3 else len(transform_functions)), math.ceil(len(transform_functions)/3), squeeze=False)
    
  i = 0
  for dr_algorithm in transform_functions.keys():
    print('==> Applying dimensionality reduction using %s on the %s dataset' % (dr_algorithm, dataset))
  
    transformer = transform_functions[dr_algorithm]
    latent_X    = transformer(original_X)
    
    # latent_X[~np.isfinite(latent_X)] = 0
  
    # Generate clusters in latent space (K-means, but could be anything)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(latent_X)
    latent_Y = kmeans.predict(latent_X)
    latent_Y_centers = kmeans.cluster_centers_
    
    # Plotting latent space
    idx1 = int(i/2 % 3)
    idx2 = int(math.floor(i/2/3))
    axs[idx1, idx2].scatter(latent_X[:, 0], latent_X[:, 1], c=latent_Y, s=2, cmap='viridis')
    centers = kmeans.cluster_centers_
    axs[idx1, idx2].scatter(latent_Y_centers[:, 0], latent_Y_centers[:, 1], c='black', s=200, alpha=0.5);
    axs[idx1, idx2].set_title(dr_algorithm)
    
    # Training and evaluating for different Ks and lambdas
    if train_algos:
      j = 0
      for k in K:
        best_measure = 0.0
        save_file_name = global_dir + "/results/deltas/" + dataset + "_" + dr_algorithm + "_" + str(variance_adjust) + "_k" + str(k) + ".npy"
        
        if load_saved_deltas and os.path.isfile(save_file_name):
          print('Loading pre-saved delta for %s, K = %d' % (dr_algorithm.upper(), k))
          
          E = Explainer(original_X, latent_Y, transformer, num_clusters, 0.5)
          E.set_delta(np.load(save_file_name))
          
          cr, cv = E.metrics(k=k)
          mean_cr = np.mean(cr)
          mean_cv = np.mean(cv)
          
          out[j, i]   = mean_cr
          out[j, i+1] = mean_cv
        else:
          for lamb in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            for trial in range(num_trials):
              E = Explainer(original_X, latent_Y, transformer, num_clusters, lamb)
              E.learn(verbose_interval=1000)
              cr, cv = E.metrics(k=k)
              
              mean_cr = np.mean(cr)
              mean_cv = np.mean(cv)
              
              print('[%s, K = %d, l = %.1f, trial %d] Average correctness: %.3f | Average coverage: %.3f' % (dr_algorithm.upper(), k, lamb, trial+1, mean_cr, mean_cv))
              
              if mean_cr > best_measure:
                best_measure = mean_cr
                out[j, i]   = mean_cr
                out[j, i+1] = mean_cv
                np.save(save_file_name, E.get_delta())
        j += 1
    i += 2 # index for DR algorithm in output matrix
    
    
  # First showing the latent space plot
  if len(transform_functions) % 2 == 1:
    axs[-1, -1].axis('off')
      
  fig.suptitle(dataset + " dataset (variance adjusted to " + str(variance_adjust) + ", " + str(num_clusters) + " clusters)")
  plt.show()
  fig.savefig(global_dir + "/results/plots/latent_spaces_" + dataset + "_" + str(datetime.timestamp(datetime.now())) + ".png", bbox_inches='tight')
  
  # Then plotting corresponding results (if training was enabled)
  if train_algos:
    results = (out, K, list(transform_functions.keys()), dataset)
    
    plot_algorithm_comparison(results)
    
    with open(global_dir + "/results/measures/" + dataset + "_" + str(datetime.timestamp(datetime.now())) + ".pickle", 'wb') as handle:
      pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  