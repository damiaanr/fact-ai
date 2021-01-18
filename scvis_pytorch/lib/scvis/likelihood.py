from torch import sum, log, sqrt, distributions
import numpy as np

EPS = 1e-20


def log_likelihood_gaussian(x, mu, sigma_square):
    return sum(-0.5 * log(2.0 * np.pi) - 0.5 * log(sigma_square) - (x - mu) ** 2 / (2.0 * sigma_square), 1)


def log_likelihood_student(x, mu, sigma_square, df=2.0):
    sigma = sqrt(sigma_square)
    dist = distributions.StudentT(df=df, loc=mu, scale=sigma)
    return sum(dist.log_prob(x), 1)
