from sklearn import decomposition
from sklearn import manifold
import numpy as np
import math
import torch
from vae.encoder import VAE

"""
We use different dimensionality reduction algorithms, which result in different
latent spaces. Each dimensionality reduction algorithm is defined by its own
'transform' function, which maps one or multiple samples X in original space
to representation/latent space.
"""


def generate_transformers(x, min_variance=10):
  """
  This function returns a dictionary with callables for a given dataset.
  """

  transform_functions = {
    'vae': (lambda x: transform_vae(x)),
    # 'pca': (lambda x: transform_pca(x, pca, var_pca)),
    # 'tsvd': (lambda x: transform_tsvd(x, tsvd)),
    # 'kpca': (lambda x: transform_kpca(x, kpca)),
    # 'spca': (lambda x: transform_spca(x, spca)),
    # 'iso': (lambda x: transform_iso(x, iso)),
    # 'lle': (lambda x: transform_lle(x, net)),
  }

  # Variational Autoencoder
  input_dim = x.shape[1]
  net = VAE(input_dim=input_dim, latent_dim=2)
  net.load_state_dict(torch.load('vae/model.pt')['model_state_dict'])
  net.eval()

  def transform_vae(x):
    x_batch = torch.from_numpy(x).float()
    encoder_mu, encoder_log_var = net.encoder(x_batch)
    batch_z = net.sampling(encoder_mu, encoder_log_var)
    return batch_z

  """
  Note that below, we could have dynamically generated most transformer
  functions. However, doing so would potentially lose overview, and we
  do not have to optimize for efficiency here, while we actually have
  to preserve readability.
  """

  # Regular PCA
  pca = decomposition.PCA(n_components=2) 
  var_pca = np.var(pca.fit_transform(x)) # We do this in one call, since we don't need latent_X for now

  def transform_pca(x, pca, var_pca):
    return np.matmul(x, np.transpose(pca.components_))/math.sqrt(var_pca)*math.sqrt(min_variance)
    

  # Truncated SVD
  tsvd = decomposition.TruncatedSVD(n_components=2, n_iter=7, random_state=42)
  var_tsvd = np.var(tsvd.fit_transform(x))

  def transform_tsvd(x, tsvd):
    return np.matmul(x, np.transpose(tsvd.components_))/math.sqrt(var_tsvd)*math.sqrt(min_variance)
    
    
  # Kernel PCA
  kpca = decomposition.KernelPCA(n_components=2, kernel="sigmoid", fit_inverse_transform=True, gamma=None, random_state=42)
  var_kpca = np.var(kpca.fit_transform(x))
  
  if 0. in kpca.lambdas_:
    del transform_functions['kpca']
  
  def transform_kpca(x, kpca):
    x = np.array(x)
    if len(x.shape) == 1:
      x = x.reshape(1,-1)
    return kpca.transform(x)/math.sqrt(var_kpca)*math.sqrt(min_variance)
    
  # Sparse PCA
  spca = decomposition.SparsePCA(n_components=2, alpha=0.0001, random_state=42, n_jobs=-1)
  var_spca = np.var(spca.fit_transform(x))
  
  def transform_spca(x, spca):
    return np.matmul(x, np.transpose(spca.components_))/math.sqrt(var_spca)*math.sqrt(min_variance)
    
  # ISO
  iso = manifold.Isomap(n_neighbors=8, n_components=2, eigen_solver='dense')
  var_iso = np.var(iso.fit_transform(x))
  
  def transform_iso(x, iso):
    x = np.array(x)
    if len(x.shape) == 1:
      x = x.reshape(1,-1)
    return iso.transform(x)/math.sqrt(var_iso)*math.sqrt(min_variance)
    
  # LLE
  lle = manifold.LocallyLinearEmbedding(n_neighbors=8, n_components=2, eigen_solver='dense')
  var_lle = np.var(lle.fit_transform(x))
  
  def transform_lle(x, lle):
    x = np.array(x)
    if len(x.shape) == 1:
      x = x.reshape(1,-1)
    return lle.transform(x)/math.sqrt(var_lle)*math.sqrt(min_variance)
  
  
  return transform_functions
  
  