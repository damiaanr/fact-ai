import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from matplotlib.legend_handler import HandlerBase

"""
Artist handlers below are used for metric plots.
"""
class LegendObjectHandlerPlots(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
                       
        l1 = plt.Line2D([x0,y0+width], [1*height,1*height],
                           c=orig_handle[0])
        l2 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], ls='--',
                           c=orig_handle[0])
        l3 = plt.Line2D([x0,y0+width], [0*height,0*height],  ls=':',
                           c=orig_handle[0])
        return [l1, l2, l3]
        
class LegendObjectHandlerMetrics(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        
        l1 = plt.Line2D([x0,y0+width], [1*height,1*height],
                           c='k')
                           
        if orig_handle[0] == 1:
          l1 = plt.Line2D([x0,y0+width], [1*height,1*height],
                             c='k', ls='--')
        elif orig_handle[0] == 2:
          l1 = plt.Line2D([x0,y0+width], [1*height,1*height],
                             c='k', ls=':')
        return [l1]

"""
Functions below are general functions used througout the project.
"""
def truncate(values, k):
  """
  Used to truncate a translation vector delta so that it becomes k-sparse
  """
  
  values = np.squeeze(values)
  idx = (-np.abs(values)).argsort()[:k]
  values_aprox = np.zeros(values.shape)
  values_aprox[idx] = values[idx]
  return values_aprox
  
def similarity(e_more, e_less):
  """
  Adapted from https://github.com/GDPlumb/ELDR/tree/master/Code, used for
  calculating the similarity measure.
  """
  
  difference = 0
  for i in range(e_more.shape[0]):
    if e_less[i] != 0:
      difference += np.abs(e_more[i])
  denom = np.sum(np.abs(e_more))
  if denom != 0:
    difference /= denom
  return difference
    
def load_data(file, skipFirstRow = False, skipFirstColumn = False, lastColumnClasses = False):
  """
  This function dynamically loads any dataset of the format:
    - one sample per line
    - features separated by spaces
  First column (sometimes ID) and first row (sometimes extra info) can be skipped
  The last column can optinally be returned as a list of class integers
  """
  
  raw = pd.read_csv(file, sep="\t").values
  
  if lastColumnClasses:
    y = raw[(1 if skipFirstRow else 0):, -1]
  else:
    y = None
  
  x = raw[(1 if skipFirstRow else 0):, (1 if skipFirstColumn else 0):(-1 if lastColumnClasses else None)]
  
  return x, y

def plot_algorithm_comparison(results, global_dir = '.', compressed_plot=True):
  """
  This function is used to generate a plot for many different dimensionality
  reduction algorithms, for all different Ks, on one dataset.
  
  if @show_compressed_plot is set to True, the legend will be compressed (newer plot)
  """
  out, K, transformers, dataset, similarity_scores = results
  
  c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  
  i = 0
  for transformer in transformers:
    plt.plot(K, out[:, i*2], c = c[i], label = "Correctness - " + transformer)
    plt.plot(K, out[:, i*2+1], ls = "--", c= c[i], label = "Coverage - " + transformer)
    plt.plot(similarity_scores[i][:,0], similarity_scores[i][:,1], ls = ":", c= c[i], label = "Similarity - " + transformer)
    
    i += 1
    
  plt.ylabel("Metric")
  plt.xlabel("Number of Features Used")
  plt.title(dataset + " dataset")
  
  if compressed_plot:
    legendMetrics = plt.legend([(0, 0), (1, 1), (2, 2)], ["Correctness", "Coverage", "Similarity"], loc=4, handler_map={tuple: LegendObjectHandlerMetrics()})
    
    plt.legend([(x, 2) for x in c[0:i]], transformers, loc=7,  
             handler_map={tuple: LegendObjectHandlerPlots()})
    plt.gca().add_artist(legendMetrics)
  else:
    plt.legend()
    
  plt.savefig(global_dir + "/results/plots/measures_" + dataset + "_" + str(datetime.timestamp(datetime.now())) + ".png", bbox_inches='tight')
  plt.show()
  
def reject_outliers(data, m=2):
  """
  This function removes outliers from a numpy array of data, and is used in
  the process of preserving big-enough variance in a dataset.
  """
  
  return data[abs(data - np.mean(data)) < m * np.std(data)]
  
def cluster_latent_space(x, num_clusters):
  """
  This function is used to generate clusters in a given latent space.
  Uses K-means, but could be anything (clustering method does not)
  matter for the explanation.
  
  Note: if a learnt explanation is re-loaded, the clusters should be
        identical all times the explanation is evaluated.
  """
  kmeans = KMeans(n_clusters=num_clusters, random_state=42)
  kmeans.fit(x)
  latent_Y = kmeans.predict(x)
  latent_Y_centers = kmeans.cluster_centers_
  
  return latent_Y, latent_Y_centers
  
def load_delta(deltas, k, initial, target):
  """
  This function is only used for calculating similarity scores
  (see main.py:261-262)
  """
  if initial == 0:
    d = deltas[target - 1]
  elif target == 0:
    d = -1.0 * deltas[initial - 1]
  else:
    d = -1.0 * deltas[initial - 1] + deltas[target - 1]
      
  d = truncate(d, k)

  return d