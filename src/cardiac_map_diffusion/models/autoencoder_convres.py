"""Residual convolutional variational autoencoder for ventricular MAP signals.

This module defines a 1D-convolutional beta-VAE with residual blocks
(``ResNetBlock``) used as a VAE baseline in the cardiac MAP denoising study: a
strided-convolution ``Encoder`` interleaved with residual blocks and BatchNorm
that emits a Gaussian latent, a mirrored transposed-convolution ``Decoder``, and
the wrapping ``AutoEncoder`` that performs the reparameterisation trick. This is a
faithful, verbatim migration of the original ``autoencoder_convres.py`` into the
package layout; the residual blocks, layer sizes, activations, and forward-pass
math are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.relu(x)
        return x

# Rest of the code remains the same...

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_size):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_size = latent_size
        self.conv1 = nn.Conv1d(1, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(2)
        self.resnet1 = ResNetBlock(2)
        self.conv2 = nn.Conv1d(2, 4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(4)
        self.resnet2 = ResNetBlock(4)
        self.conv3 = nn.Conv1d(4, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.resnet3 = ResNetBlock(16)
        self.conv4 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.resnet4 = ResNetBlock(32)
        self.conv5 = nn.Conv1d(32, 64, kernel_size=3, stride=5, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.resnet5 = ResNetBlock(64)
        self.conv6 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(32)
        self.resnet6 = ResNetBlock(32)
        self.fc_mean = nn.Linear(37 * 32, latent_size)
        self.fc_log_var = nn.Linear(37 * 32, latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resnet1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.resnet2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.resnet3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.resnet4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.resnet5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.resnet6(x)

        x = torch.flatten(x, 1)
        mean_x = self.fc_mean(x)
        logvar_x = self.fc_log_var(x)
        return mean_x, logvar_x

# Rest of the code remains the same...

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_size):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 32 * 37)
        self.conv1 = nn.ConvTranspose1d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.resnet1 = ResNetBlock(64)
        self.conv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=5, output_padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.resnet2 = ResNetBlock(32)
        self.conv3 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.resnet3 = ResNetBlock(16)
        self.conv4 = nn.ConvTranspose1d(16, 4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(4)
        self.resnet4 = ResNetBlock(4)
        self.conv5 = nn.ConvTranspose1d(4, 2, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(2)
        self.resnet5 = ResNetBlock(2)
        self.conv6 = nn.ConvTranspose1d(2, 1, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(1)
        self.resnet6 = ResNetBlock(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.reshape(x, (-1, 32, 37))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resnet1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.resnet2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.resnet3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.resnet4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.resnet5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.resnet6(x)

        x = torch.squeeze(x)
        return x
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, latent_size)
        self.decoder = Decoder(input_dim, latent_size)
    def forward(self, x):
        # encode x into latent distr
        mean_x, logvar_x = self.encoder(x)
        sigma = torch.exp(logvar_x / 2)
        # get sample from latent
        z = mean_x + sigma * torch.randn_like(mean_x)
        # decode sample from latent
        recon_x = self.decoder(z)
        return recon_x, mean_x, logvar_x

#input = torch.randn((16,370))
#model = AutoEncoder(input_dim=370, latent_size=32)

#output = model(input)
#pdb