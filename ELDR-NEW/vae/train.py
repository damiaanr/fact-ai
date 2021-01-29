import torch
import numpy as np
from vae.dataset import DataSet
from vae.encoder import VAE, LATENT_DIMENSION, LEARNING_RATE, BATCH_SIZE, MAX_EPOCH, CustomLoss, PERPLEXITY, L2_REGULARISATION
import datetime

MIN_ITER = 3000
MAX_ITER = 30000
CLIP_VALUE = 3.0
CLIP_NORM = 10.0
torch.manual_seed(42)

def trainVAE(x, global_dir, dataset):
  train_data = DataSet(x)

  input_dim = x.shape[1]

  # Neural net object, optimizer and criterion
  net = VAE(input_dim, latent_dim=LATENT_DIMENSION)
  net.train()
  optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=0.001, weight_decay=L2_REGULARISATION)
  criterion = CustomLoss(input_dim, net)

  # Run a training epoch
  iter_per_epoch = round(x.shape[0] / BATCH_SIZE)
  iter_n = int(iter_per_epoch * MAX_EPOCH)
  if iter_n < MIN_ITER:
      iter_n = MIN_ITER
  elif iter_n > MAX_ITER:
      iter_n = np.max([MAX_ITER, iter_per_epoch * 2])

  print("Started training VAE on " + dataset + ": "+ str(datetime.datetime.now().time()))
  print("Total iterations: " + str(iter_n))
  for iter_i in range(iter_n):
      if iter_i % 50 == 0:
          print("Iter: " + '{:>8}'.format(str(iter_i)) + "   " + str(datetime.datetime.now().time()))

      x_batch = train_data.next_batch(BATCH_SIZE)

      # clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
      optimizer.zero_grad()

      p, z, encoder_mu, encoder_log_var, decoder_mu, decoder_log_var, dof = net(x_batch)
      loss = criterion(x_batch=x_batch,
                       p=p,
                       z_batch=z,
                       encoder_mu=encoder_mu,
                       encoder_log_var=encoder_log_var,
                       decoder_mu=decoder_mu,
                       decoder_log_var=decoder_log_var,
                       iter=iter_i+1,
                       dof=dof)

      # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
      loss.backward()

      # Clip gradients like in SCVIS to prevent exploding gradients
      torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=CLIP_NORM)
      for name, var in net.named_parameters():
          if 'encoder_layer_sigma_square' in name:
              torch.clamp(var.grad, -CLIP_VALUE*0.1, CLIP_VALUE*0.1)
          else:
              torch.clamp(var.grad, -CLIP_VALUE, CLIP_VALUE)


      # optimizer.step() causes the optimizer to take a step based on the gradients of the parameters.
      optimizer.step()
  print("Ended training VAE: "+ str(datetime.datetime.now().time()))

  torch.save({
      'perplexity': PERPLEXITY,
      'regularizer': L2_REGULARISATION,
      'batch_size': BATCH_SIZE,
      'learning_rate': LEARNING_RATE,
      'latent_dimension': LATENT_DIMENSION,
      'activation': 'ELU',
      'model_state_dict': net.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
  }, global_dir + "/results/vae_models/" + dataset + ".pt")