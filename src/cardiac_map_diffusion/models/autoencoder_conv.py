"""Convolutional (CNN) variational autoencoder for ventricular MAP signals.

This module defines the 1D-convolutional beta-VAE used as a VAE baseline in the
cardiac MAP denoising study. It provides several selectable encoder/decoder
``architecture`` variants (0-6) built from ``Conv1d``/``ConvTranspose1d`` stacks
with LeakyReLU activations, plus the wrapping ``AutoEncoder`` that performs the
reparameterisation trick. This is a faithful, verbatim migration of the original
``autoencoder_conv.py`` into the package layout; all architecture branches, layer
sizes, activations, weight scaling, and forward-pass math are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_size, architecture=5, weight_constant=1):
        #super().__init__()
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_size = latent_size

        if architecture == 0:
            self.fc1 = nn.Conv1d(1, 128, 3, padding="same") # N, 128, 370
            self.fc2 = nn.Conv1d(128, 64, 3, padding="same") # N, 64, 370
            self.fc3 = nn.Conv1d(64, 32, 3, padding="same") # N, 32, 370
            self.fc_mean = nn.Linear(32 * self.input_dim, latent_size)
            self.fc_log_var = nn.Linear(32 * self.input_dim, latent_size)
        elif architecture == 1:
            self.fc1 = nn.Conv1d(1, 16, 3, padding="same") # N, 16, 370
            self.fc2 = nn.Conv1d(16, 4, 3, padding="same") # N, 4, 370
            self.fc3 = nn.Conv1d(4, 2, 3, padding="same") # N, 2, 370
            self.fc_mean = nn.Linear(2 * self.input_dim, latent_size)
            self.fc_log_var = nn.Linear(2 * self.input_dim, latent_size)
        elif architecture == 2:
            self.fc1 = nn.Conv1d(1, 16, kernel_size=3, padding=1) # N, 16, 370
            self.fc2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1) # N, 32, 185
            self.fc3 = nn.Conv1d(32, 64, kernel_size=3, stride=5) # N, 64, 37
            self.fc4 = nn.Conv1d(64, 128, kernel_size=37) # N, 128, 1
            self.fc_mean = nn.Linear(128 * 1, latent_size)
            self.fc_log_var = nn.Linear(128 * 1, latent_size)
        elif architecture == 3:
            self.fc1 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1) # N, 1, 185
            self.fc2 = nn.Conv1d(1, 1, kernel_size=3, stride=5, padding=1) # N, 1, 37
            self.fc_mean = nn.Linear(37 * 1, latent_size)
            self.fc_log_var = nn.Linear(37 * 1, latent_size)
        elif architecture == 4:
            self.fc1 = nn.Conv1d(1, 2, kernel_size=3, padding="same") # N, 2, 370
            self.fc2 = nn.Conv1d(2, 4, kernel_size=3, padding="same") # N, 4, 370
            self.fc3 = nn.Conv1d(4, 16, kernel_size=3, stride=2, padding=1) # N, 16, 185
            self.fc4 = nn.Conv1d(16, 32, kernel_size=3, stride=5, padding=1) # N, 32, 37
            self.fc_mean = nn.Linear(37 * 32, latent_size)
            self.fc_log_var = nn.Linear(37 * 32, latent_size)
        elif architecture == 5:

            self.fc1 = nn.Conv1d(1, 2, kernel_size=3, padding="same")  # N, 2, 370
            self.fc2 = nn.Conv1d(2, 4, kernel_size=3, padding="same")  # N, 4, 370
            self.fc3 = nn.Conv1d(4, 16, kernel_size=3, padding="same") # N, 16, 370
            self.fc4 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)  # N, 32, 185
            self.fc5 = nn.Conv1d(32, 64, kernel_size=3, stride=5, padding=1)  # N, 64, 37
            self.fc6 = nn.Conv1d(64, 32, kernel_size=3, padding=1) # N, 32, 37
            self.fc_mean = nn.Linear(37 * 32, latent_size)
            self.fc_log_var = nn.Linear(37 * 32, latent_size)
            '''
            self.fc1 = nn.Conv1d(1, 2, kernel_size=3, padding="same")
            self.bn1 = nn.BatchNorm1d(2)
            self.fc2 = nn.Conv1d(2, 4, kernel_size=3, padding="same")
            self.bn2 = nn.BatchNorm1d(4)
            self.fc3 = nn.Conv1d(4, 16, kernel_size=3, padding="same")
            self.bn3 = nn.BatchNorm1d(16)
            self.fc4 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
            self.bn4 = nn.BatchNorm1d(32)
            self.fc5 = nn.Conv1d(32, 64, kernel_size=3, stride=5, padding=1)
            self.bn5 = nn.BatchNorm1d(64)
            self.fc6 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm1d(32)
            self.fc_mean = nn.Linear(37 * 32, latent_size)
            self.fc_log_var = nn.Linear(37 * 32, latent_size)
            '''
        elif architecture == 6:
            self.fc1 = nn.Conv1d(1, 4, kernel_size=3, padding="same")  # N, 4, 370
            self.fc2 = nn.Conv1d(4, 16, kernel_size=3, padding="same") # N, 16, 370
            self.fc3 = nn.Conv1d(16, 64, kernel_size=3, padding="same") # N, 64, 370
            self.fc4 = nn.Conv1d(64, 32, kernel_size=3, padding=1)  # N, 32, 370
            self.fc5 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1) # N, 32, 185
            self.fc6 = nn.Conv1d(32, 32, kernel_size=3, stride=5, padding=1)  # N, 32, 37
            self.fc_mean = nn.Linear(37 * 32, latent_size)
            self.fc_log_var = nn.Linear(37 * 32, latent_size)

        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        #self.relu = nn.PReLU()
        #self.relu = nn.ELU()
        #self.relu = nn.SiLU()
        #self.relu = nn.Hardswish

        # Multiply weights by constant

        for module in self.modules():
            if isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight *= weight_constant

        #self.init_weights()

    #Initialization
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)

    def forward(self, x, architecture=5):
        x = torch.unsqueeze(x, 1)
        if architecture == 0 or architecture == 1:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
        elif architecture == 2 or architecture == 4:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.relu(x)
        elif architecture == 3:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
        elif architecture==5 or architecture == 6:

            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.relu(x)
            x = self.fc5(x)
            x = self.relu(x)
            x = self.fc6(x)
            x = self.relu(x)
            '''
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.fc5(x)
            x = self.bn5(x)
            x = self.relu(x)
            x = self.fc6(x)
            x = self.bn6(x)
            x = self.relu(x)
            '''

        # ReLu
        # flatten before dense layer
        x = torch.flatten(x, 1)
        mean_x = self.fc_mean(x)
        logvar_x = self.fc_log_var(x)
        return mean_x, logvar_x

    def print_initial_weights(self):
        print("Encoder Initial Weights:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_size, architecture=5, weight_constant=1):
        #super().__init__()
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_size = latent_size

        if architecture == 0:
            self.fc1 = nn.Linear(latent_size, 32 * self.input_dim)
            self.fc2 = nn.Conv1d(32, 64, 3, padding="same")
            self.fc3 = nn.Conv1d(64, 128, 3, padding="same")
            self.fc4 = nn.Conv1d(128, 1, 3, padding="same")
        elif architecture == 1:
            self.fc1 = nn.Linear(latent_size, 2 * self.input_dim)
            self.fc2 = nn.ConvTranspose1d(2, 4, 3, padding=3)
            self.fc3 = nn.ConvTranspose1d(4, 16, 3)
            self.fc4 = nn.ConvTranspose1d(16, 1, 3)
        elif architecture == 2:
            self.fc1 = nn.Linear(latent_size, 128)
            self.fc2 = nn.ConvTranspose1d(128, 64, kernel_size=37)
            self.fc3 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=5, output_padding=2)
            self.fc4 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.fc5 = nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1)
        elif architecture == 3:
            self.fc1 = nn.Linear(latent_size, 37) # N, 37
            self.fc2 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=5, output_padding=2) # N, 1, 185
            self.fc3 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # N, 1, 370
        elif architecture == 4:
            self.fc1 = nn.Linear(latent_size, 32*37) # N, 32*37
            self.fc2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=5, output_padding=2) # N, 16, 185
            self.fc3 = nn.ConvTranspose1d(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1) # N, 4, 370
            self.fc4 = nn.ConvTranspose1d(4, 2, kernel_size=3, padding=1) # N, 2, 370
            self.fc5 = nn.ConvTranspose1d(2, 1, kernel_size=3, padding=1) # N, 1, 370
        elif architecture == 5:

            self.fc1 = nn.Linear(latent_size, 32*37) # N, 32*37
            self.fc2 = nn.ConvTranspose1d(32, 64, kernel_size=3, padding=1) # N, 64, 37
            self.fc3 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=5, output_padding=2) # N, 32, 185
            self.fc4 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1) # N, 16, 370
            self.fc5 = nn.ConvTranspose1d(16, 4, kernel_size=3, padding=1) # N, 4, 370
            self.fc6 = nn.ConvTranspose1d(4, 2, kernel_size=3, padding=1) # N, 2, 370
            self.fc7 = nn.ConvTranspose1d(2, 1, kernel_size=3, padding=1) # N, 1, 370
            '''
            self.fc1 = nn.Linear(latent_size, 32 * 37)
            self.bn1 = nn.BatchNorm1d(32)
            self.fc2 = nn.ConvTranspose1d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc3 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=5, padding=1, output_padding=1)
            self.bn3 = nn.BatchNorm1d(32)
            self.fc4 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.bn4 = nn.BatchNorm1d(16)
            self.fc5 = nn.ConvTranspose1d(16, 8, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm1d(8)
            self.fc6 = nn.ConvTranspose1d(8, 4, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm1d(4)
            self.fc7 = nn.ConvTranspose1d(4, 1, kernel_size=7, padding=0)
            '''
        elif architecture == 6:
            self.fc1 = nn.Linear(latent_size, 32 * 37)  # N, 32*37
            self.fc2 = nn.ConvTranspose1d(32, 32, kernel_size=3, stride=5, output_padding=2)  # N, 32, 185
            self.fc3 = nn.ConvTranspose1d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # N, 32, 370
            self.fc4 = nn.ConvTranspose1d(32, 64, kernel_size=3, padding=1)  # N, 64, 370
            self.fc5 = nn.ConvTranspose1d(64, 16, kernel_size=3, padding=1)  # N, 16, 370
            self.fc6 = nn.ConvTranspose1d(16, 4, kernel_size=3, padding=1)  # N, 4, 370
            self.fc7 = nn.ConvTranspose1d(4, 1, kernel_size=3, padding=1)  # N, 1, 370

        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        #self.relu = nn.PReLU()
        #self.relu = nn.ELU()
        #self.relu = nn.SiLU()
        #self.relu = nn.Hardswish

        # Multiply weights by constant
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight *= weight_constant

        #self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)

    def forward(self, x, architecture=5):

        if architecture == 0:
            x = self.fc1(x)
            x = self.relu(x)
            x = torch.reshape(x, (-1, 32, self.input_dim))
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.relu(x)
        elif architecture == 1:
            x = self.fc1(x)
            x = self.relu(x)
            x = torch.reshape(x, (-1, 2, self.input_dim))
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.relu(x)
        elif architecture == 2:
            x = self.fc1(x)
            x = self.relu(x)
            x = torch.reshape(x, (-1, 128, 1))
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.relu(x)
            x = self.fc5(x)
            x = self.relu(x)
        elif architecture == 3:
            x = self.fc1(x)
            x = self.relu(x)
            x = torch.reshape(x, (-1, 1, 37))
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
        elif architecture == 4:
            x = self.fc1(x)
            x = self.relu(x)
            x = torch.reshape(x, (-1, 32, 37))
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.relu(x)
            x = self.fc5(x)
            x = self.relu(x)
        elif architecture == 5 or architecture == 6:

            x = self.fc1(x)
            x = self.relu(x)
            x = torch.reshape(x, (-1, 32, 37))
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.relu(x)
            x = self.fc5(x)
            x = self.relu(x)
            x = self.fc6(x)
            x = self.relu(x)
            x = self.fc7(x)
            '''
            x = self.fc1(x)
            x = self.relu(x)
            x = torch.reshape(x, (-1, 32, 37))
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.fc5(x)
            x = self.bn5(x)
            x = self.relu(x)
            x = self.fc6(x)
            x = self.bn6(x)
            x = self.relu(x)
            x = self.fc7(x)
            #x = self.bn6(x)
            #x = self.relu(x)
            '''

        x = torch.squeeze(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_size, architecture=5, weight_constant=1):
        super().__init__()
        self.latent_size = latent_size
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, latent_size, architecture, weight_constant)
        self.decoder = Decoder(input_dim, latent_size, architecture, weight_constant)
    def forward(self, x, architecture=5):
        # encode x into latent distr
        mean_x, logvar_x = self.encoder(x, architecture)
        sigma = torch.exp(logvar_x / 2)
        # get sample from latent
        z = mean_x + sigma * torch.randn_like(mean_x)
        # decode sample from latent
        recon_x = self.decoder(z, architecture)
        return recon_x, mean_x, logvar_x
