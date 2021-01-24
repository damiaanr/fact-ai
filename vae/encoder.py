import torch
from torch import nn, sqrt, distributions, Tensor
from torch.nn import functional as F
import pandas as pd
import numpy as np
from scvis import data
from scvis.tsne_helper import compute_transition_probability

EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10
LEARNING_RATE = 0.01
L2_REGULARISATION = 0.001
MAX_EPOCH = 100
BATCH_SIZE = 505
PERPLEXITY = 10
LATENT_DIMENSION = 2

def log_likelihood_student(x, mu, sigma_square, df=2.0):
    sigma = sqrt(sigma_square)
    dist = distributions.StudentT(df=df, loc=mu, scale=sigma)
    return torch.sum(dist.log_prob(x), dim=1)

def init_w_b(layer):
    nn.init.xavier_normal_(layer.weight, gain=1.0)
    nn.init.constant_(layer.bias, 0.1)

class VAE(nn.Module):
    """
    @see https://medium.com/analytics-vidhya/complete-guide-to-build-an-autoencoder-in-pytorch-and-keras-94555dce395d
    """
    def __init__(self, input_dim: int, latent_dim: int):
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

    def encoder(self, x_batch):
        h1 = F.elu(self.encoder_layer1(x_batch))
        h2 = F.elu(self.encoder_layer2(h1))
        h3 = F.elu(self.encoder_layer3(h2))

        mu = self.encoder_layer_mu(h3)
        log_var = self.encoder_layer_sigma_square(h3)
        log_var = torch.clamp(F.softplus(log_var), EPS, MAX_SIGMA_SQUARE)

        # In SCVIS vae.py mu is computed using the output of the MLP and dropout is used on the weights tensor
        # I am not sure whether computing mu first and then do F.dropout works the same
        mu = F.dropout(mu, p=0.9)

        return mu, log_var

    def sampling(self, encoder_mu, encoder_log_var, batch_size=BATCH_SIZE):
        ep = torch.randn([batch_size, self._latent_dim])
        latent_z = torch.add(encoder_mu, torch.sqrt(encoder_log_var) * ep)
        return latent_z


    def decoder(self, z):
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

        # TODO Is this the right moment to clamp the DOF tensor?
        dof = torch.clamp(self.dof, 0.1, 10)

        return p, latent_z, encoder_mu, encoder_log_var, decoder_mu, decoder_log_var, dof


class CustomLoss(nn.Module):
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
