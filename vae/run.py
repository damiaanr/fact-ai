import torch
import pandas as pd
import numpy as np
from scvis import data
from vae.encoder import VAE, LATENT_DIMENSION, LEARNING_RATE, BATCH_SIZE, MAX_EPOCH, CustomLoss, PERPLEXITY, L2_REGULARISATION


# Script for housing
x = pd.read_csv('./Housing/Data/X.tsv', sep='\t').values
y = pd.read_csv('./Housing/Data/y.tsv', sep='\t').values
train_data = data.DataSet(x, y)

input_dim = x.shape[1]
net = VAE(input_dim, latent_dim=LATENT_DIMENSION)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=0.001, weight_decay=L2_REGULARISATION)

iter_per_epoch = round(x.shape[0] / BATCH_SIZE)
max_iter = int(iter_per_epoch * MAX_EPOCH)
if max_iter < 3000:
    max_iter = 3000
elif max_iter > 30000:
    max_iter = np.max([30000, iter_per_epoch * 2])

criterion = CustomLoss(input_dim, net)
for iter_i in range(max_iter):
    x_batch, y_batch = train_data.next_batch(BATCH_SIZE)

    # clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
    optimizer.zero_grad()

    y_predicted, p, z, encoder_mu, encoder_log_var, decoder_mu, decoder_log_var, dof = net(x_batch, iter_i + 1)
    loss = criterion(y_predicted,
                     p=p,
                     z=z,
                     encoder_mu=encoder_mu,
                     encoder_log_var=encoder_log_var,
                     decoder_mu=decoder_mu,
                     decoder_log_var=decoder_log_var,
                     iter=iter_i+1)

    # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
    loss.backward()

    # TODO clip_gradient like in scvis
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)

    # optimizer.step() causes the optimizer to take a step based on the gradients of the parameters.
    optimizer.step()

torch.save({
    'perplexity': PERPLEXITY,
    'regularizer': L2_REGULARISATION,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'latent_dimension': LATENT_DIMENSION,
    'activation': 'ELU',
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'model.pt')