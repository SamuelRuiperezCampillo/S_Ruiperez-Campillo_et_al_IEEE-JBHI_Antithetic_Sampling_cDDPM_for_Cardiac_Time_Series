"""Fully-connected (FC) variational autoencoder for ventricular MAP signals.

This module defines a simple fully-connected beta-VAE used as one of the VAE
baselines in the cardiac MAP denoising study: a three-layer ReLU ``Encoder`` that
emits a Gaussian latent (mean / log-variance), a mirrored three-layer ``Decoder``,
and the wrapping ``AutoEncoder`` that performs the reparameterisation trick. This
is a faithful, verbatim migration of the original ``autoencoder.py`` module into
the package layout; the network definition, layer sizes, activations, and
forward-pass math are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_size):
        super().__init__()
        self.input_dim = input_dim
        self.latent_size = latent_size
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

        self.fc_mean = nn.Linear(32, latent_size)
        self.fc_log_var = nn.Linear(32, latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        # ReLu
        mean_x = self.fc_mean(x)
        logvar_x = self.fc_log_var(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_size):
        super().__init__()
        self.input_dim = input_dim
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, latent_size)
        self.decoder = Decoder(input_dim, latent_size)

    def forward(self, x, is_training=True):
        # encode x into latent distr
        mean_x, logvar_x = self.encoder(x)
        sigma = torch.exp(logvar_x / 2)
        # get sample from latent
        if is_training:
            z = mean_x + sigma * torch.randn_like(mean_x)
        else:
            z = mean_x # for the test set
        # decode sample from latent
        recon_x = self.decoder(z)
        return recon_x, mean_x, logvar_x
