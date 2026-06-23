''' Faithful migration of the original `denoising_net_small.py` (the smaller DeScoD-ECG variant with 3 instead of 5 HNF blocks per stream). The architecture and ConditionalModelSmall definition are reproduced verbatim; no imports required rewriting and no filesystem paths were present, so only this migration note was prepended to the existing module docstring.

    - This is the denoising network introduced in DeScoD-ECG (DOI: 10.1109/JBHI.2023.3237712).

    - The number of HNF blocks is reduced from 5 to 3 to have fewer parameters. 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log as ln

# Custom Convolutional Layer with He Initialization
class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initializes weights with a He normal initializer, suitable for ReLUs
        nn.init.kaiming_normal_(self.weight)
        # Initializes biases to zero
        nn.init.zeros_(self.bias)

# Positional Encoding Layer for Injecting Noise Level Information
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # Dimension of the positional encoding

    def forward(self, noise_level):
        # Reshape noise level input for processing
        noise_level = noise_level.view(-1)
        # Half of the dimension size to generate sinusoidal encodings
        count = self.dim // 2
        # Generate steps for sinusoidal calculation
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        # Calculate encoding using the exponential decay and trigonometric functions
        encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        #print(encoding)
        '''
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.dim, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        '''
        return encoding

# Feature Wise Affine Transformation Layer for Conditional Scaling and Shifting
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        # Linear layer to generate parameters for affine transformation
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            # Split the output of the linear layer into gamma and beta for scaling and shifting
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            # Apply only shifting without scaling
            x = x + self.noise_func(noise_embed).view(batch, -1, 1)
        return x

# Hierarchical Noise Filtering Block with Multiple Dilated Convolutions
class HNFBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        # Multiple dilated convolutions to capture features at various scales
        self.filters = nn.ModuleList([
            Conv1d(input_size, hidden_size//4, 3, dilation=dilation, padding=1*dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size//4, 5, dilation=dilation, padding=2*dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size//4, 9, dilation=dilation, padding=4*dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size//4, 15, dilation=dilation, padding=7*dilation, padding_mode='reflect'),
        ])
        # Convolution to combine the outputs of the dilated filters
        self.conv_1 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')
        # Instance normalization to stabilize the learning by normalizing the activations
        self.norm = nn.InstanceNorm1d(hidden_size//2)
        # Final convolution to refine the output after combining dilated filters
        self.conv_2 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')
        
    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # Apply each dilated filter and store their outputs
        filts = []
        for layer in self.filters:
            filts.append(layer(x))
            
        # Concatenate filter outputs and split into two groups
        filts = torch.cat(filts, dim=1)
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        
        # Apply activation function and normalize one part of the split output
        filts = F.leaky_relu(torch.cat([self.norm(nfilts), filts], dim=1), 0.2)
        
        # Apply final convolution and activation, then add the residual
        filts = F.leaky_relu(self.conv_2(filts), 0.2)
        
        return filts + residual
        
# Bridge Layer for Processing and Combining Noise Embedding with Signal
class Bridge(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Feature-wise affine transformation to condition the input based on noise
        self.encoding = FeatureWiseAffine(input_size, hidden_size, use_affine_level=1)
        # Convolutional layers to process the input signal
        self.input_conv = Conv1d(input_size, input_size, 3, padding=1, padding_mode='reflect')
        self.output_conv = Conv1d(input_size, hidden_size, 3, padding=1, padding_mode='reflect')
    
    def forward(self, x, noise_embed):
        # Process input with convolution, apply noise conditioning, then refine output with another convolution
        x = self.input_conv(x)
        x = self.encoding(x, noise_embed)
        return self.output_conv(x)
    

# Main Conditional Model for Denoising
class ConditionalModelSmall(nn.Module):
    def __init__(self, feats=64):
        super(ConditionalModelSmall, self).__init__()
        # Define two streams of processing: one for the input signal and one for conditioning
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'), nn.LeakyReLU(0.2)),
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4)
        ])
        
        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'), nn.LeakyReLU(0.2)),
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4)
        ])
        
        # Positional encoding for noise level
        self.embed = PositionalEncoding(feats)
        
        # Bridges to combine the outputs of the two streams
        self.bridge = nn.ModuleList([
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats)
        ])
        
        # Final convolutional layer to output the denoised signal
        self.conv_out = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')
        
    def forward(self, x, cond, noise_scale):
        # Embed the noise scale into a positional encoding
        noise_embed = self.embed(noise_scale)
        xs = []
        # Process the input signal through each layer in the stream and apply the bridge
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            #print(x.shape)
            xs.append(br(x, noise_embed))
        
        # Combine the processed signal with the conditioning stream
        for x, layer in zip(xs, self.stream_cond):
            #print(cond.shape)
            cond = layer(cond) + x
        
        # Output the denoised signal
        return self.conv_out(cond)

