import torch
from torch import nn, distributions, add, matmul, clamp
from scvis.tf_helper import weight_xavier_relu, bias_variable, shape
from collections import namedtuple
import tensorflow as tf

LAYER_SIZE = [128, 64, 32]
OUTPUT_DIM = 2
KEEP_PROB = 1.0
EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10

LocationScale = namedtuple('LocationScale', ['mu', 'sigma_square'])


# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, batch_size, layer_size, output_dim,
                 activate_op=tf.nn.elu,
                 init_w_op=weight_xavier_relu,
                 init_b_op=bias_variable):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.input_size = batch_size

        self.layer_size = layer_size
        self.output_dim = output_dim
        self.num_encoder_layer = len(self.layer_size)

        self._activate, self.init_w, self.init_b = activate_op(), init_w_op, init_b_op

        # Create weights and biases for neural net
        self.weights = [self.init_w([self.input_dim, layer_size[0]])]
        self.biases = [self.init_b([layer_size[0]])]
        for in_dim, out_dim in zip(layer_size, layer_size[1:]):
            self.weights.append(self.init_w([in_dim, out_dim]))
            self.biases.append(self.init_b([out_dim]))

    def _forward(self, input_data):
        """
        Pass input through the hidden layers
        :return:
        """
        weights, biases = self.weights, self.biases
        layer_size = self.layer_size

        hidden_layer_out = torch.matmul(input_data, weights[-1]) + biases[-1]
        hidden_layer_out = self._activate(hidden_layer_out)

        for in_dim, out_dim in zip(layer_size, layer_size[1:]):
            hidden_layer_out = self._activate(matmul(hidden_layer_out, weights[-1]) + biases[-1])

        return hidden_layer_out

class GaussianEncoder(MLP):
    def __init__(self, input_dim, batch_size,
                 layer_size=LAYER_SIZE,
                 output_dim=OUTPUT_DIM,
                 decoder_layer_size=LAYER_SIZE[::-1],
                 prod = 0.9):
        super(GaussianEncoder, self).__init__(input_dim, batch_size, layer_size, output_dim)

        self._prob = prod
        self._dropout = torch.nn.Dropout(p=self.prod)
        self._softplus = torch.nn.Softplus()

        # Encoder variables
        self.bias_mu = self.init_b([self.output_dim])
        self.weights_mu = self.init_w([self.layer_size[-1], self.output_dim])

        self.bias_sigma_square = self.init_b([self.output_dim])
        self.weights_sigma_square = self.init_w([self.layer_size[-1], self.output_dim])

        self.encoder_parameter = self.encoder()

        # Variables for running 'sample'
        self.ep = torch.normal(0, 1, [self.input_size, self.output_dim])  # TF name was 'epsilon_univariate_norm'
        self.z = torch.add(self.encoder_parameter.mu,
                           torch.sqrt(self.encoder_parameter.sigma_square) * self.ep)  # TF name was 'latent_z'

        self.decoder_layer_size = decoder_layer_size
        self.num_decoder_layer = len(self.decoder_layer_size)

    def encoder(self):
        weights_mu = self._dropout(self.weights_mu)
        output = self._forward(weights_mu)
        mu = add(matmul(output, weights_mu), self.bias_mu)
        sigma_square = add(matmul(self.hidden_layer_out, self.weights_sigma_square), self.bias_sigma_square)

        return LocationScale(mu, clamp(self._softplus(sigma_square), EPS, MAX_SIGMA_SQUARE))

    def forward(self, input_data):
        return self._forward(input_data)


class GaussianDecoder(MLP):
    def __init__(self, input_dim, batch_size,
                 layer_size=LAYER_SIZE,
                 output_dim=OUTPUT_DIM,
                 decoder_layer_size=LAYER_SIZE[::-1]):
        super(GaussianDecoder, self).__init__(input_dim, batch_size, layer_size, output_dim)

        self.decoder_layer_size = decoder_layer_size
        self.num_decoder_layer = len(self.decoder_layer_size)

        self._append_weights_and_biases()

    def decoder(self, z):
        hidden_layer_out = self.activate(
            matmul(z, self.weights[self.num_encoder_layer]) +
            self.biases[self.num_encoder_layer]
        )

        for layer in range(self.num_encoder_layer + 1, self.num_encoder_layer + self.num_decoder_layer):
            hidden_layer_out = self.activate(
                matmul(hidden_layer_out, self.weights[layer]) +
                self.biases[layer]
            )

        mu = add(matmul(hidden_layer_out, self.decoder_weights_mu), self.decoder_bias_mu)
        sigma_square = tf.add(tf.matmul(hidden_layer_out, self.decoder_weights_sigma_square),
                              self.decoder_bias_sigma_square
                              )
        return LocationScale(mu,
                             torch.clamp(tf.nn.softplus(sigma_square), EPS, MAX_SIGMA_SQUARE))

    def forward(self, output_vector):
        """
        The decoder starts with the Net output vector and goes backwards through the network (network = weights and biases).

        :param output_vector:
        :return:
        """
        decoder_hidden_layer_out = self.activate(matmul(output_vector, self.weights[-1]) + self.biases[-1])
        for in_dim, out_dim in zip(self.decoder_layer_size, self.decoder_layer_size[1:]):
            decoder_hidden_layer_out = self.activate(matmul(decoder_hidden_layer_out, self.weights[-1])
                                                     + self.biases[-1])

    def _append_weights_and_biases(self):
        """
        When instantiating the decoder, more weight and biases are required. Furthermore, additional
        instance variables need to be set.
        :return:
        """
        self.weights.append(self.init_w([self.output_dim, self.decoder_layer_size[0]]))
        self.biases.append(self.init_b([self.decoder_layer_size[0]]))
        self.decoder_hidden_layer_out = self.activate(matmul(self.z, self.weights[-1]) + self.biases[-1])

        for in_dim, out_dim in zip(self.decoder_layer_size, self.decoder_layer_size[1:]):
            self.weights.append(self.init_w([in_dim, out_dim]))
            self.biases.append(self.init_b([out_dim]))
            self.decoder_hidden_layer_out = self.activate(
                matmul(self.decoder_hidden_layer_out, self.weights[-1]) +
                self.biases[-1])

        self.decoder_bias_mu = self.init_b([self.input_dim])
        self.decoder_weights_mu = self.init_w([self.decoder_layer_size[-1],self.input_dim])

        self.decoder_bias_sigma_square = self.init_b([self.input_dim])
        self.decoder_weights_sigma_square = self.init_w([self.decoder_layer_size[-1], self.input_dim])

        mu = add(matmul(self.decoder_hidden_layer_out, self.decoder_weights_mu),
                 self.decoder_bias_mu)
        sigma_square = add(matmul(self.decoder_hidden_layer_out, self.decoder_weights_sigma_square),
                           self.decoder_bias_sigma_square)

        self.decoder_parameter = LocationScale(mu, clamp(tf.nn.softplus(sigma_square), EPS, MAX_SIGMA_SQUARE))
