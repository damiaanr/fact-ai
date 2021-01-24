import csv
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances
from scipy import optimize
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from functions import truncate

class Explainer:
  lamb = 0.5 # regularization term lambda
  stop_iters    = 2000 # if no improve in loss, stop after X iters
  minimum_iters = 2000 # except if not a minimum of Y iters has been done
  consecutive_steps = 10 # in negative gradient direction per sampling
  loss_discount = 0.99
  tol = 0.0001

  def __init__(self, X, Y, transformer, num_clusters, lamb = 0.5, global_dir = '.'):
    # unlabeled points in input space
    self.original_X = X
    
    # cluster labels in latent space, indices corresponding to X
    self.latent_Y   = Y
    self.num_clusters = num_clusters
    
    # 'r' function (dimension map)
    self.transformer      = transformer
    
    # for saving plots
    self.global_dir = global_dir
    
    # prepare
    self.set_lambda(lamb)
    self.calculate_latent_X()
    self.calculate_original_Y()
    
  def set_lambda(self, lamb):
    self.lamb = lamb
    
  def get_delta(self):
    return self.delta
    
  def set_delta(self, delta):
    """
    If this method is used, learn() is not necessary.
    """
    
    self.delta = delta
    
  def calculate_latent_X(self):
    self.latent_X = self.transformer(self.original_X)
    
  def calculate_original_Y(self):
    """
    Labels samples in input space according to their labels in latent space.
    Also calculates means of the clusters, both in input and latent space.
    """
    
    num_samples  = self.original_X.shape[0]
    
    # Labels in input space (back-reversing, inspired by compressed sensing)
    self.original_Y     = -1.0 * np.ones((num_samples))
    
    # Means of clusters in both spaces
    self.original_means = [[]] * self.num_clusters
    self.latent_means   = [[]] * self.num_clusters
    
    # Only used as temporary indexing variable
    indices             = [[]] * self.num_clusters
    
    for i in range(self.num_clusters):
      indices[i] = []
      
      for j in range(num_samples):
        if self.latent_Y[j] == i:
          indices[i].append(j)
          self.original_Y[j] = i
      
      self.original_means[i] = np.mean(self.original_X[indices[i], :], axis = 0)
      self.latent_means[i]   = np.mean(self.latent_X[indices[i], :], axis = 0)
    
  def loss(self, delta, cluster_initial, cluster_target):
    # translation of the initial cluster in the original space
    cluster_initial_translated = self.original_means[cluster_initial] + delta
    
    # mapping of translated initial cluster to latent space
    cluster_initial_projected  = self.transformer(cluster_initial_translated)
  
    # euclidean distance between projection and target (goal) in latent space
    latent_distance = cluster_initial_projected - self.latent_means[cluster_target]
    
    regularization_term = self.lamb * np.linalg.norm(delta)
    
    return np.linalg.norm(latent_distance, ord=2)**2 + regularization_term
    
  def learn(self, verbose_interval = 100, learning_rate = 0.0005):
    num_dimensions = self.original_X.shape[1]
    eps = np.sqrt(np.finfo(float).eps) # delta range for optimizator
  
    # contains the explanations (zero cluster is reference group)
    deltas = best_deltas = np.zeros((self.num_clusters - 1, num_dimensions))
  
    i = best_i = 0
    loss = best_loss = np.inf
    ema = None
    
    while True:
      if i - best_i > self.stop_iters and i > self.minimum_iters:
        break

      # random sample of two clusters
      if i % self.consecutive_steps == 0:
        cluster_initial, cluster_target = np.random.choice(self.num_clusters, 2, replace = False)
        
      if cluster_initial == 0:
        d = deltas[cluster_target - 1]
      elif cluster_target == 0:
        d = -1.0 * deltas[cluster_initial - 1]
      else:
        d = -1.0 * deltas[cluster_initial - 1] + deltas[cluster_target - 1]
        
      current_loss = self.loss(d, cluster_initial, cluster_target) # not efficient...
      gradient_delta = optimize.approx_fprime(d, self.loss, [eps] * num_dimensions, cluster_initial, cluster_target)
      
      if i == 0:
        ema = current_loss
      else:
        ema = self.loss_discount * ema + (1 - self.loss_discount) * current_loss
        
      if ema < best_loss - self.tol:
        best_i = i
        best_loss = ema
        best_deltas = deltas
        
      if i % verbose_interval == 0:
        print('Current loss: %.3f | best loss: %.3f (iter %d) | currently comparing %d -> %d' % (ema, best_loss, i, cluster_initial, cluster_target))
        
      gradient_delta = np.clip(np.squeeze(gradient_delta), -1.0 * 5.0, 5.0)
        
      if cluster_initial == 0:
        deltas[cluster_target - 1] -= learning_rate * gradient_delta
      elif cluster_target == 0:
        deltas[cluster_initial - 1] += learning_rate * gradient_delta
      else:
        deltas[cluster_initial - 1] += learning_rate * 0.5 * gradient_delta
        deltas[cluster_target - 1] -= learning_rate * 0.5 * gradient_delta
        
      i += 1
      
    print('Done training!')
    
    self.delta = best_deltas
    
  def metrics(self, epsilon = 1.0, k = None):
    num_dimensions = self.original_X.shape[1]

    correctness = np.zeros((self.num_clusters, self.num_clusters))
    coverage = np.zeros((self.num_clusters, self.num_clusters))
    
    for initial in range(self.num_clusters):
      for target in range(self.num_clusters):
        x_init   = np.array([self.original_X[i] for i in range(len(self.original_X)) if self.original_Y[i] == initial])
        x_target = np.array([self.original_X[i] for i in range(len(self.original_X)) if self.original_Y[i] == target])
        
        # Construct the explanation between the initial and target regions
        if initial == target:
          d = np.zeros((1, num_dimensions))
        elif initial == 0:
          d = self.delta[target - 1]
        elif target == 0:
          d = -1.0 * self.delta[initial - 1]
        else:
          d = -1.0 * self.delta[initial - 1] + self.delta[target - 1]
            
        if k is not None:
          d = truncate(d, k)
        
        r_init   = self.transformer(x_init + d)
        r_target = self.transformer(x_target)
        
        dists = euclidean_distances(r_init, Y = r_target)
        
        close_enough = 1.0 * (dists <= epsilon)
        
        if initial == target:
          threshold = 2.0
        else:
          threshold = 1.0

        correctness[initial, target] = np.mean(1.0 * (np.sum(close_enough, axis = 1) >= threshold))
        coverage[initial, target]    = np.mean(1.0 * (np.sum(close_enough, axis = 0) >= threshold))

    self.correctness = correctness
    self.coverage    = coverage

    return correctness, coverage
   
  def plot_metrics(self, fontsize = 55, labelsize = 40):
    # Set up figure and image grid
    fig = plt.figure(figsize=(20, 10))
        
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1,2),
                     axes_pad=0.75,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.25,
                     )

    # Add data to image grid
    a = self.correctness
    b = self.coverage
    c = 0
    
    for ax in grid:
      ax.tick_params(axis = "both", which = "major", labelsize = labelsize)
      
      if c == 0:
        im = ax.imshow(a, cmap = "RdYlGn", interpolation = "none", vmin = 0.0, vmax = 1.0)
        ax.set_title("Correctness - " + str(np.round(np.mean(a), 3)), fontsize = fontsize)
        ax.set_ylabel("Initial Group", fontsize = fontsize)
      elif c == 1:
        im = ax.imshow(b, cmap = "RdYlGn", interpolation = "none", vmin = 0.0, vmax = 1.0)
        ax.set_title("Coverage - "  + str(np.round(np.mean(b), 3)), fontsize = fontsize)
      ax.set_xlabel("Target Group", fontsize = fontsize)
      c += 1

    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    ax.cax.tick_params(labelsize = labelsize)

    plt.savefig(self.global_dir + "/results/plots/metrics_plot_" + str(datetime.timestamp(datetime.now())) + ".png")
    plt.show()
    plt.close()
    
  def plot_explanation(self, c1, c2, k = None, num_points = 50, feature_names = None):
    # Find the explanation from c1 to c2
    if c1 == 0:
      d = self.delta[c2 - 1]
    elif c2 == 0:
      d = -1.0 * self.delta[c1 - 1]
    else:
      d = -1.0 * self.delta[c1 - 1] + self.delta[c2 - 1]
    
    if k is not None:
      d = truncate(d, k)
        
    a = self.correctness
    b = self.coverage
    d = np.reshape(d, (1, d.shape[0]))
   
    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 30))
    fig.subplots_adjust(hspace = .3)

    for i in range(2):
      if i == 0:
        initial = c1
        target = c2
        sign = 1.0
      elif i == 1:
        initial = c2
        target = c1
        sign = -1.0

      # Plot the full representation
      ax = plt.subplot(3, 1, i + 1)
      
      plt.scatter(self.latent_X[:, 0], self.latent_X[:, 1])
  
      # Sample num_points in initial group
      # indices_initial = np.random.choice(indices[initial], num_points, replace = False)
      # points_initial = x[indices_initial, :]
      
      points_initial = np.array([self.original_X[i] for i in range(len(self.original_X)) if self.original_Y[i] == initial])
      
      #print(points_initial.shape)
      
      if(points_initial.shape[0] < num_points):
        num_points = points_initial.shape[0]
      
      points_initial = points_initial[np.random.choice(points_initial.shape[0], num_points, replace=False), :]
  
      # Plot the chosen points before perturbing them
      y_initial = self.transformer(points_initial)
      plt.scatter(y_initial[:,0], y_initial[:,1], marker = "v", c = "magenta")
  
      # Plot the chosen points after perturbing them
      y_after = self.transformer(points_initial + sign * d)
      plt.scatter(y_after[:,0], y_after[:,1], marker = "v", c = "red")
  
      plt.title("Mapping from Group " + str(initial) + " to Group " + str(target) + "\n Correctness - " + str(np.round(a[initial, target], 3)) + ", Coverage - " + str(np.round(b[initial, target], 3)))
  
    ax = plt.subplot(3, 1, 3)

    feature_index = np.array(range(d.shape[1]))
    plt.scatter(feature_index, d)
    plt.title("Explanation for Group " + str(c1) + " to Group " + str(c2))
    plt.ylabel("Change applied")
    if feature_names is None:
        plt.xlabel("Feature Index")
    else:
        plt.xlabel("Feature")
        plt.xticks(range(d.shape[1]), feature_names, rotation=90, fontsize = 40)

    plt.savefig(self.global_dir + "/results/plots/explanation_plot_" + str(datetime.timestamp(datetime.now())) + ".png")
    plt.show()
    plt.close()
