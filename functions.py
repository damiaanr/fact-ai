import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def truncate(values, k):
  """
  Used to truncate a translation vector delta so that it becomes k-sparse
  """
  
  values = np.squeeze(values)
  idx = (-np.abs(values)).argsort()[:k]
  values_aprox = np.zeros(values.shape)
  values_aprox[idx] = values[idx]
  return values_aprox
    
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

def plot_algorithm_comparison(results, global_dir = '.'):
  out, K, transformers, dataset = results
  
  c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  
  i = 0
  for transformer in transformers:
    plt.plot(K, out[:, i*2], c = c[i], label = "Correctness - " + transformer)
    plt.plot(K, out[:, i*2+1], ls = "--", c= c[i], label = "Coverage - " + transformer)
    
    i += 1
    
  plt.ylabel("Metric")
  plt.xlabel("Number of Features Used")
  plt.title(dataset + " dataset")
  plt.legend()
  plt.savefig(global_dir + "/results/plots/measures_" + dataset + "_" + str(datetime.timestamp(datetime.now())) + ".png", bbox_inches='tight')
  plt.show()
  
def reject_outliers(data, m=2):
  """
  This function removes outliers from a numpy array of data, and is used in
  the process of preserving big-enough variance in a dataset.
  """
  
  return data[abs(data - np.mean(data)) < m * np.std(data)]