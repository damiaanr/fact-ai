from sklearn import decomposition
from sklearn import manifold
import numpy as np
import math
import torch
import os.path
from vae.train import trainVAE
from vae.dataset import DataSet
from vae.encoder import VAE, LATENT_DIMENSION, LEARNING_RATE, BATCH_SIZE, MAX_EPOCH, CustomLoss, PERPLEXITY, L2_REGULARISATION
import datetime

torch.manual_seed(42)

"""
We use different dimensionality reduction algorithms, which result in different
latent spaces. Each dimensionality reduction algorithm is defined by its own
'transform' function, which maps one or multiple samples X in original space
to representation/latent space.
"""

def generate_transformers(x, dataset, global_dir, min_variance=10, additional_scale_tsvd = 1):
  """
  This function returns a dictionary with callables for a given dataset.
  """

  transform_functions = {
    'vae': (lambda x: transform_vae(x, VAE_net)),
    'pca': (lambda x: transform_pca(x, pca, var_pca)),
    'tsvd': (lambda x: transform_tsvd(x, tsvd)),
    'kpca': (lambda x: transform_kpca(x, kpca)),
    'spca': (lambda x: transform_spca(x, spca)),
    'iso': (lambda x: transform_iso(x, iso)),
    'lle': (lambda x: transform_lle(x, lle)),
  }


  """
  Note that below, we could have dynamically generated most transformer
  functions. However, doing so would potentially lose overview, and we
  do not have to optimize for efficiency here, while we actually have
  to preserve readability.
  """

  ################ Regular PCA ################
  
  pca = decomposition.PCA(n_components=2) 
  var_pca = np.var(pca.fit_transform(x)) # We do this in one call, since we don't need latent_X for now

  def transform_pca(x, pca, var_pca):
    return np.matmul(x, np.transpose(pca.components_))/math.sqrt(var_pca)*math.sqrt(min_variance)
    




  ################ Truncated SVD ################
  tsvd = decomposition.TruncatedSVD(n_components=2, n_iter=7, random_state=42)
  var_tsvd = np.var(tsvd.fit_transform(x))

  def transform_tsvd(x, tsvd):
    return np.matmul(x, np.transpose(tsvd.components_))/math.sqrt(var_tsvd)*math.sqrt(min_variance)*additional_scale_tsvd
    
    
    
    
    
    
  ################ Kernel PCA ################
  kpca = decomposition.KernelPCA(n_components=2, kernel="sigmoid", fit_inverse_transform=True, gamma=None, random_state=42)
  var_kpca = np.var(kpca.fit_transform(x))
  
  if 0. in kpca.lambdas_: # KPCA with Sigmoid kernel does not work for this set
    del transform_functions['kpca']
  
  def transform_kpca(x, kpca):
    x = np.array(x)
    if len(x.shape) == 1:
      x = x.reshape(1,-1)
    return kpca.transform(x)/math.sqrt(var_kpca)*math.sqrt(min_variance)
    
    
    
    
    
    
  ################ Sparse PCA ################
  spca = decomposition.SparsePCA(n_components=2, alpha=0.0001, random_state=42, n_jobs=-1)
  var_spca = np.var(spca.fit_transform(x))
  
  def transform_spca(x, spca):
    return np.matmul(x, np.transpose(spca.components_))/math.sqrt(var_spca)*math.sqrt(min_variance)
    
    
    
    
    
    
  ################ ISO ################
  iso = manifold.Isomap(n_neighbors=8, n_components=2, eigen_solver='dense')
  var_iso = np.var(iso.fit_transform(x))
  
  def transform_iso(x, iso):
    x = np.array(x)
    if len(x.shape) == 1:
      x = x.reshape(1,-1)
    return iso.transform(x)/math.sqrt(var_iso)*math.sqrt(min_variance)
    
    
    
    
    
    
  ################ LLE ################
  lle = manifold.LocallyLinearEmbedding(n_neighbors=8, n_components=2, eigen_solver='dense')
  var_lle = np.var(lle.fit_transform(x))
  
  def transform_lle(x, lle):
    x = np.array(x)
    if len(x.shape) == 1:
      x = x.reshape(1,-1)
    return lle.transform(x)/math.sqrt(var_lle)*math.sqrt(min_variance)
  
  
  
  
  
  
  ################ SCVIS VAE ################
  VAE_save_file = global_dir + "/results/vae_models/" + dataset + ".pt"
  
  if not os.path.isfile(VAE_save_file):
    # Auto-encoder needs to be trained on the model first
    print('Training new VAE model on %s dataset' % dataset)
    trainVAE(x, global_dir, dataset) # normalizing using np.max(np.abs(x)) not necessary as it equals 1
  
  # Once trained, it loads existing model, also for reproducability
  VAE_model = torch.load(VAE_save_file)['model_state_dict']
  
  print('Loaded VAE model for %s dataset' % dataset)
    
  VAE_net = VAE(input_dim=x.shape[1], latent_dim=2)
  VAE_net.load_state_dict(VAE_model)
  VAE_net.eval()
  
  def transform_vae(x, VAE_net):
    x = np.array(x)
    if len(x.shape) == 1:
      x = x.reshape(1,-1)
      
    with torch.no_grad():
      x_batch = torch.from_numpy(x).float()
      encoder_mu, encoder_log_var = VAE_net.encoder(x_batch, p=1.0)
      batch_z = VAE_net.sampling(encoder_mu, encoder_log_var, batch_size=len(x), eval=True).numpy()

    return np.array(batch_z, dtype=float)
  
  return transform_functions
  