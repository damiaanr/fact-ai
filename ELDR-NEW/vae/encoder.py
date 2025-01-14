import torch
from torch import nn, sqrt, distributions, Tensor
from torch.nn import functional as F
import numpy as np
import sys

MAX_VAL = np.log(sys.float_info.max) / 2.0

# Hyperparameters used for training the VAE
EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10
LEARNING_RATE = 0.01
L2_REGULARISATION = 0.001
MAX_EPOCH = 100
BATCH_SIZE = 100
PERPLEXITY = 10
LATENT_DIMENSION = 2

torch.manual_seed(42)

def log_likelihood_student(x, mu, sigma_square, df=2.0):
    sigma = sqrt(sigma_square)
    dist = distributions.StudentT(df=df, loc=mu, scale=sigma)
    return torch.sum(dist.log_prob(x), dim=1)

def init_w_b(layer):
    nn.init.xavier_normal_(layer.weight, gain=1.0)
    nn.init.constant_(layer.bias, 0.1)

class VAE(nn.Module):
    """
    Gaussian Variational AutoEncoder based on the paper "Interpretable dimensionality reduction of single cell
    transcriptome data with deep generative model" by Ding et al.
    (https://www.nature.com/articles/s41467-018-04368-5)

    This autoencoder is a PyTorch implementation of the authors' Tensorflow implementation
    (https://github.com/shahcompbio/scvis.)
    """
    def __init__(self, input_dim: int, latent_dim: int):
        """
        @param input_dim: Number of features of each data point
        @param latent_dim:  Number of latent dimensions
        """
        super(VAE, self).__init__()

        self._input_dim = input_dim
        self._latent_dim = latent_dim

        # Encoder layers
        self.encoder_layer1 = nn.Linear(self._input_dim, 128)
        init_w_b(self.encoder_layer1)

        self.encoder_layer2 = nn.Linear(128, 64)
        init_w_b(self.encoder_layer2)

        self.encoder_layer3 = nn.Linear(64, 32)
        init_w_b(self.encoder_layer3)

        self.encoder_layer_mu = nn.Linear(32, latent_dim)
        init_w_b(self.encoder_layer_mu)

        self.encoder_layer_sigma_square = nn.Linear(32, self._latent_dim)
        init_w_b(self.encoder_layer_sigma_square)

        # Decoder layers
        self.decoder_layer1 = nn.Linear(self._latent_dim, 32)
        init_w_b(self.decoder_layer1)

        self.decoder_layer2 = nn.Linear(32, 32)
        init_w_b(self.decoder_layer2)

        self.decoder_layer3 = nn.Linear(32, 32)
        init_w_b(self.decoder_layer3)

        self.decoder_layer4 = nn.Linear(32, 64)
        init_w_b(self.decoder_layer4)

        self.decoder_layer5 = nn.Linear(64, 128)
        init_w_b(self.decoder_layer5)

        self.decoder_layer_mu = nn.Linear(128, self._input_dim)
        init_w_b(self.decoder_layer_mu)

        self.decoder_layer_sigma_square = nn.Linear(128, self._input_dim)
        init_w_b(self.decoder_layer_sigma_square)

        dof_tensor = torch.ones(size=[self._input_dim], dtype=torch.float32)
        self.dof = nn.Parameter(dof_tensor, requires_grad=True)  # requires_grad=True to make it trainable

    def encoder(self, x_batch, p=0.9):
        """
        @param x_batch: A batch of data points to be processed
        @param p: Probability to keep a data point in the Dropout layer. The Dropout is used during training a model.
        It is recommended that during inference, this value should be 1.0, i.e. keep all data points.
        @return:
        """
        h1 = F.elu(self.encoder_layer1(x_batch))
        h2 = F.elu(self.encoder_layer2(h1))
        h3 = F.elu(self.encoder_layer3(h2))

        weights_mu = F.dropout(self.encoder_layer_mu.weight, p=1.0 - p)

        mu = torch.add(torch.matmul(h3, torch.transpose(weights_mu, 0, 1)), self.encoder_layer_mu.bias)
        
        log_var = self.encoder_layer_sigma_square(h3)
        log_var = torch.clamp(F.softplus(log_var), EPS, MAX_SIGMA_SQUARE)

        return mu, log_var

    def sampling(self, encoder_mu, encoder_log_var, batch_size=BATCH_SIZE, eval = False):
        """
        @param encoder_mu: Mu returned by this class its encoder function
        @param encoder_log_var: Variance returned by this class its encoder function
        @param batch_size:
        @param eval:
        @return:
        """
        if eval:
          ep = 0.5  # we keep the points static during inference, so gradients can successfully find a direction
        else:
          ep = torch.randn([batch_size, self._latent_dim])  # but not during training of the model
          
        latent_z = torch.add(encoder_mu, torch.sqrt(encoder_log_var) * ep)
        return latent_z


    def decoder(self, z):
        """
        @param z: A batch of data points mapped into the latent space.
        @return:
        """
        h1 = F.elu(self.decoder_layer1(z))
        h2 = F.elu(self.decoder_layer2(h1))
        h3 = F.elu(self.decoder_layer3(h2))
        h4 = F.elu(self.decoder_layer4(h3))
        h5 = F.elu(self.decoder_layer5(h4))

        mu = self.decoder_layer_mu(h5)
        sigma_square_before_clamped = self.decoder_layer_sigma_square(h5)
        sigma_square = torch.clamp(F.softplus(sigma_square_before_clamped), EPS, MAX_SIGMA_SQUARE)

        return mu, sigma_square

    def forward(self, x_batch: np.array):
        p = compute_transition_probability(x_batch, perplexity=PERPLEXITY)
        x_batch = torch.from_numpy(x_batch)
        encoder_mu, encoder_log_var = self.encoder(x_batch)
        latent_z = self.sampling(encoder_mu, encoder_log_var)
        decoder_mu, decoder_log_var = self.decoder(latent_z)

        dof = torch.clamp(self.dof, 0.1, 10)

        return p, latent_z, encoder_mu, encoder_log_var, decoder_mu, decoder_log_var, dof


class CustomLoss(nn.Module):
    """
    Custom loss used in combination with the VAE from the paper "Interpretable dimensionality reduction of single cell
    transcriptome data with deep generative model" by Ding et al.
    (https://www.nature.com/articles/s41467-018-04368-5)
    """
    def __init__(self, input_dim: int, net):
        super(CustomLoss, self).__init__()
        self._input_dim = input_dim
        self._net = net

    def forward(self,
                x_batch,
                p,
                z_batch,
                encoder_mu,
                encoder_log_var,
                decoder_mu,
                decoder_log_var,
                iter: int,
                dof: Tensor):

        elbo = self._elbo(x_batch=x_batch,
                          p=p,
                          iter=iter,
                          encoder_mu=encoder_mu,
                          encoder_log_var=encoder_log_var,
                          decoder_mu=decoder_mu,
                          decoder_log_var=decoder_log_var,
                          dof=dof)
        kl_pq = self._tsne_repel(z_batch=z_batch, p=p) * np.min([iter, self._input_dim])
        l2_regularisation = self._l2_regulariser()
        objective = kl_pq + l2_regularisation - elbo

        return objective

    def _l2_regulariser(self):
        # Computes half the L2 norm of a tensor without the `sqrt`: output = sum(t ** 2) / 2
        # Converted from tf.nn.l2_loss
        penalty = [torch.sum(var ** 2) / 2 for name, var in self._net.named_parameters() if 'weight' in name]
        l2_regularizer = L2_REGULARISATION * sum(penalty)
        return l2_regularizer

    def _tsne_repel(self, z_batch, p):
        nu = LATENT_DIMENSION

        sum_y = torch.sum(torch.square(z_batch), dim=1)
        matmul_result = torch.matmul(z_batch, torch.transpose(z_batch, 0, 1))
        num = -2.0 * matmul_result + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / nu

        p_out = torch.from_numpy(p) + 0.1 / BATCH_SIZE
        p_out = p_out / torch.unsqueeze(torch.sum(p_out, dim=1), 1)

        num = torch.pow(1.0 + num, -(nu + 1.0) / 2.0)
        attraction = torch.multiply(p_out, torch.log(num))
        attraction = -torch.sum(attraction)

        den = torch.sum(num, dim=1) - 1
        repellant = torch.sum(torch.log(den))

        return (repellant + attraction) / BATCH_SIZE

    def _elbo(self, x_batch, p, iter, encoder_mu, encoder_log_var, decoder_mu, decoder_log_var, dof):  # Compute ELBO: Evidence Lower Bound
        p = torch.from_numpy(p)
        x_batch = torch.from_numpy(x_batch)
        weights = torch.clamp(torch.sum(p, 0), 0.01, 2.0)

        # Compute log likelihood
        lls_result = log_likelihood_student(x_batch, decoder_mu, decoder_log_var, dof)
        multiplication_res = torch.multiply(lls_result, weights)
        log_likelihood = torch.mean(multiplication_res)

        # Compute KL divergence
        kl_divergence = torch.mean(0.5 * torch.sum(encoder_mu ** 2 +
                                                   encoder_log_var -
                                                   torch.log(encoder_log_var) - 1,
                                                   dim=1))
        kl_divergence *= np.max([0.1, self._input_dim / iter])

        elbo = log_likelihood - kl_divergence
        return elbo

def compute_entropy(dist=np.array([]), beta=1.0):
    """
    Original function from https://github.com/shahcompbio/scvis/lib/scvis/tsne_helper.py
    """
    p = -dist * beta
    shift = MAX_VAL - max(p)
    p = np.exp(p + shift)
    sum_p = np.sum(p)

    h = np.log(sum_p) - shift + beta * np.sum(np.multiply(dist, p)) / sum_p

    return h, p / sum_p


def compute_transition_probability(x, perplexity=30.0,
                                   tol=1e-4, max_iter=50, verbose=False):
    # x should be properly scaled so the distances are not either too small or too large

    if verbose:
        print('tSNE: searching for sigma ...')

    (n, d) = x.shape
    sum_x = np.sum(np.square(x), 1)

    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    p = np.zeros((n, n))

    # Parameterized by precision
    beta = np.ones((n, 1))
    entropy = np.log(perplexity) / np.log(2)

    # Binary search for sigma_i
    idx = range(n)
    for i in range(n):
        idx_i = list(idx[:i]) + list(idx[i+1:n])

        beta_min = -np.inf
        beta_max = np.inf

        # Remove d_ii
        dist_i = dist[i, idx_i]
        h_i, p_i = compute_entropy(dist_i, beta[i])
        h_diff = h_i - entropy

        iter_i = 0
        while np.abs(h_diff) > tol and iter_i < max_iter:
            if h_diff > 0:
                beta_min = beta[i].copy()
                if np.isfinite(beta_max):
                    beta[i] = (beta[i] + beta_max) / 2.0
                else:
                    beta[i] *= 2.0
            else:
                beta_max = beta[i].copy()
                if np.isfinite(beta_min):
                    beta[i] = (beta[i] + beta_min) / 2.0
                else:
                    beta[i] /= 2.0

            h_i, p_i = compute_entropy(dist_i, beta[i])
            h_diff = h_i - entropy

            iter_i += 1

        p[i, idx_i] = p_i

    if verbose:
        print('Min of sigma square: {}'.format(np.min(1 / beta)))
        print('Max of sigma square: {}'.format(np.max(1 / beta)))
        print('Mean of sigma square: {}'.format(np.mean(1 / beta)))

    return p