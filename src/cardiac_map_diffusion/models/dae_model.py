"""Denoising AutoEncoder (DAE) baseline for cardiac MAP signal denoising.

This module defines a 2D-convolutional Denoising AutoEncoder ported from a
TensorFlow reference implementation: a strided ``Conv2d`` encoder compressing the
signal to a ``z_dim`` latent (via a sizing linear layer) and a mirrored
``ConvTranspose2d`` decoder, with LeakyReLU activations and a final linear layer
that restores the exact ``input_dim``. This is a faithful, verbatim migration of
the original ``dae_model.py`` into the package layout; the dynamic shape
inference, layer sizes, activations, weight initialisation, and forward/loss math
are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

z_dim = 32
input_dim = 1024


def _same_pad(kernel_h):
    # approximate "SAME" padding for even/odd kernel heights (symmetric)
    return kernel_h // 2


class DAE(nn.Module):
    def __init__(self, z_dim=z_dim, input_dim=input_dim):
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        
        print(f"[DAE] Initializing DAE with z_dim={z_dim}, input_dim={input_dim}")

        # Define activation first (needed for helper methods)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Encoder (mirror of your TF cnn sequence)
        # Input to encoder: (batch, 1, input_dim, 1)
        kh = 16
        pad = _same_pad(kh)
        self.enc_conv1 = nn.Conv2d(1, 40, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0))
        self.enc_bn1 = nn.BatchNorm2d(40)

        self.enc_conv2 = nn.Conv2d(40, 20, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0))
        self.enc_bn2 = nn.BatchNorm2d(20)

        self.enc_conv3 = nn.Conv2d(20, 20, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0))
        self.enc_bn3 = nn.BatchNorm2d(20)

        self.enc_conv4 = nn.Conv2d(20, 20, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0))
        self.enc_bn4 = nn.BatchNorm2d(20)

        self.enc_conv5 = nn.Conv2d(20, 40, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0))
        self.enc_bn5 = nn.BatchNorm2d(40)

        # final conv produces 1 channel and preserves stride (1,1)
        self.enc_conv6 = nn.Conv2d(40, 1, kernel_size=(kh, 1), stride=(1, 1), padding=(pad, 0))

        # Calculate the actual output size by running a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim, 1)
            dummy_output = self._forward_conv_layers(dummy_input)
            self.encoder_output_size = dummy_output.numel()  # Total number of elements
            self.conv_output_shape = dummy_output.shape[2:]  # (height, width) - excluding batch and channel dims
        
        # Add a linear layer to ensure we get exactly z_dim dimensions
        self.enc_fc = nn.Linear(self.encoder_output_size, z_dim)

        # Decoder starts with a linear layer to expand z_dim to encoder_output_size
        self.dec_fc = nn.Linear(z_dim, self.encoder_output_size)

        # Decoder (transposed convs to invert the encoder)
        # Input to decoder: (batch, 1, encoder_output_size, 1)
        self.dec_tconv1 = nn.ConvTranspose2d(1, 1, kernel_size=(kh, 1), stride=(1, 1), padding=(pad, 0))
        self.dec_bn1 = nn.BatchNorm2d(1)

        self.dec_tconv2 = nn.ConvTranspose2d(1, 40, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0), output_padding=(1, 0))
        self.dec_bn2 = nn.BatchNorm2d(40)

        self.dec_tconv3 = nn.ConvTranspose2d(40, 20, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0), output_padding=(1, 0))
        self.dec_bn3 = nn.BatchNorm2d(20)

        self.dec_tconv4 = nn.ConvTranspose2d(20, 20, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0), output_padding=(1, 0))
        self.dec_bn4 = nn.BatchNorm2d(20)

        self.dec_tconv5 = nn.ConvTranspose2d(20, 20, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0), output_padding=(1, 0))
        self.dec_bn5 = nn.BatchNorm2d(20)

        self.dec_tconv6 = nn.ConvTranspose2d(20, 40, kernel_size=(kh, 1), stride=(2, 1), padding=(pad, 0), output_padding=(1, 0))
        self.dec_bn6 = nn.BatchNorm2d(40)

        # final conv to get single output channel (same as TF conv2d conv7)
        self.dec_conv7 = nn.Conv2d(40, 1, kernel_size=(kh, 1), stride=(1, 1), padding=(pad, 0))

        # Test the decoder output size and add final linear layer for exact size match
        with torch.no_grad():
            dummy_latent = torch.zeros(1, z_dim)
            dummy_decode_out = self._test_decoder_size(dummy_latent)
            self.decoder_conv_output_size = dummy_decode_out.numel()
            
        # Add final linear layer to ensure exact input_dim match
        self.dec_final_fc = nn.Linear(self.decoder_conv_output_size, input_dim)

        # initialize weights
        self._init_weights()
        print(f"[DAE] Model initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def _forward_conv_layers(self, x):
        """Helper method to compute the output of just the conv layers"""
        h = self.act(self.enc_bn1(self.enc_conv1(x)))
        h = self.act(self.enc_bn2(self.enc_conv2(h)))
        h = self.act(self.enc_bn3(self.enc_conv3(h)))
        h = self.act(self.enc_bn4(self.enc_conv4(h)))
        h = self.act(self.enc_bn5(self.enc_conv5(h)))
        h = self.enc_conv6(h)  # last conv, no activation
        return h

    def _test_decoder_size(self, z):
        """Helper method to test decoder conv output size (before final FC layer)"""
        # Expand z_dim to encoder_output_size using linear layer
        h = self.dec_fc(z)  # (B, z_dim) -> (B, encoder_output_size)
        
        # Reshape to 4D for transposed convolutions using the stored conv output shape
        h = h.view(h.size(0), 1, *self.conv_output_shape)  # (B, 1, height, width)

        h = self.act(self.dec_bn1(self.dec_tconv1(h)))
        h = self.act(self.dec_bn2(self.dec_tconv2(h)))
        h = self.act(self.dec_bn3(self.dec_tconv3(h)))
        h = self.act(self.dec_bn4(self.dec_tconv4(h)))
        h = self.act(self.dec_bn5(self.dec_tconv5(h)))
        h = self.act(self.dec_bn6(self.dec_tconv6(h)))
        h = self.dec_conv7(h)  # (B,1,conv_output_dim,1)
        h = h.view(h.size(0), -1)  # (B, conv_output_dim)
        # NOTE: We don't apply dec_final_fc here since we want to measure the conv output size
        return h

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        """
        x: tensor shape (batch, input_dim) or (batch, input_dim, 1, 1)
        returns: z (batch, z_dim)
        """
        # Ensure input is float32 to match model weights
        x = x.float()
        
        if x.dim() == 2:
            x = x.view(x.size(0), self.input_dim, 1, 1)  # (B, H, W, C) in TF -> PyTorch needs (B, C, H, W)
            # current shape (B, H, W, C) -> convert to (B, C, H, W)
            # but above view gives (B, H, W, 1) so swap axes:
            x = x.permute(0, 3, 1, 2)  # (B, 1, input_dim, 1)
        elif x.dim() == 4 and x.size(1) == self.input_dim:
            # if user passed (B, input_dim, 1, 1) (TF-like), convert to (B, 1, input_dim, 1)
            x = x.permute(0, 3, 1, 2)

        # Forward through conv layers
        h = self._forward_conv_layers(x)
        # h shape: (B, 1, encoder_output_size, 1)
        h = h.view(h.size(0), -1)  # -> (B, encoder_output_size)
        
        # Apply linear layer to get exactly z_dim dimensions
        h = self.enc_fc(h)  # -> (B, z_dim)
        return h

    def decode(self, z):
        """
        z: tensor shape (batch, z_dim)
        returns: reconstruction (batch, input_dim)
        """
        # Ensure input is float32 to match model weights
        z = z.float()
        
        # Expand z_dim to encoder_output_size using linear layer
        h = self.dec_fc(z)  # (B, z_dim) -> (B, encoder_output_size)
        
        # Reshape to 4D for transposed convolutions using the stored conv output shape
        h = h.view(h.size(0), 1, *self.conv_output_shape)  # (B, 1, height, width)

        h = self.act(self.dec_bn1(self.dec_tconv1(h)))
        h = self.act(self.dec_bn2(self.dec_tconv2(h)))
        h = self.act(self.dec_bn3(self.dec_tconv3(h)))
        h = self.act(self.dec_bn4(self.dec_tconv4(h)))
        h = self.act(self.dec_bn5(self.dec_tconv5(h)))
        h = self.act(self.dec_bn6(self.dec_tconv6(h)))
        h = self.dec_conv7(h)  # (B,1,conv_output_dim,1)
        h = h.view(h.size(0), -1)  # (B, conv_output_dim)
        
        # Apply final linear layer to get exactly input_dim dimensions
        h = self.dec_final_fc(h)  # (B, conv_output_dim) -> (B, input_dim)
        return h

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def loss(self, x, training=True):
        """
        x: (batch, input_dim) expected
        returns: dict with 'mse' loss (scalar tensor)
        """
        # Ensure input is float32
        x = x.float()
        
        if x.dim() != 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
        x_recon, z = self.forward(x_flat)
        # Ensure both tensors are float32 for MSE computation
        mse = torch.nn.functional.mse_loss(x_recon.float(), x_flat.float(), reduction='mean')
        return {'mse': mse}