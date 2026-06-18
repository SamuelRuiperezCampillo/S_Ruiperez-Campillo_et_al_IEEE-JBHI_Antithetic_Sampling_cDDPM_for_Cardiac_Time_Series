"""LUNet (lightweight U-Net) baseline for cardiac MAP signal denoising.

This module defines a 1D U-Net denoiser built from depthwise-separable
convolution blocks (``DepthwiseSeparableConv1d`` / ``DeepSepConvBlock``) and group
convolutions (``GroupConvBlock``) with skip connections, a fixed-resolution
``LUNet``, an ``AdaptiveLUNet`` that adjusts pooling/upsampling to arbitrary input
lengths, and a ``LUNetModel`` factory. This is a faithful, verbatim migration of
the original ``lunet_model.py`` into the package layout; the block definitions,
layer sizes, pooling factors, activations, weight initialisation, and
forward/loss math are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise Separable Convolution = Depthwise Conv + Pointwise Conv"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=False):
        super().__init__()
        # Depthwise convolution (groups=in_channels means each input channel is convolved separately)
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                 padding=(kernel_size-1)//2 * dilation, dilation=dilation, 
                                 groups=in_channels, bias=bias)
        # Pointwise convolution (1x1 conv to combine channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DeepSepConvBlock(nn.Module):
    """Depthwise Separable Convolution Block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GroupConvBlock(nn.Module):
    """Group Convolution Block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=8):
        super().__init__()
        # Ensure groups divides both in_channels and out_channels
        groups = min(groups, in_channels, out_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=(kernel_size-1)//2 * dilation, dilation=dilation,
                            groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LUNet(nn.Module):
    """
    LUNet implementation based on the architecture table.
    This is a U-Net with skip connections for 1D signal denoising.
    """
    def __init__(self, input_dim=3600):
        super().__init__()
        self.input_dim = input_dim
        
        print(f"[LUNet] Initializing LUNet with input_dim={input_dim}")
        
        # Encoder (Downsampling path)
        self.enc_block1 = DeepSepConvBlock(1, 16, kernel_size=7, dilation=2)     # [1,16,3600]
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)                       # [1,16,720]
        
        self.enc_block2 = DeepSepConvBlock(16, 32, kernel_size=5, dilation=2)    # [1,32,720]
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)                       # [1,32,360]
        
        self.enc_block3 = DeepSepConvBlock(32, 64, kernel_size=3, dilation=2)    # [1,64,360]
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)                       # [1,64,180]
        
        # Bottleneck
        self.bottleneck = DeepSepConvBlock(64, 64, kernel_size=3, dilation=2)    # [1,64,180]
        
        # Decoder (Upsampling path)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')             # [1,64,360]
        self.dec_block1 = GroupConvBlock(64+64, 32, kernel_size=3, dilation=2)   # [1,32,360] (64+64 due to skip connection)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')             # [1,32,720]
        self.dec_block2 = GroupConvBlock(32+32, 16, kernel_size=5, dilation=2)   # [1,16,720] (32+32 due to skip connection)
        
        self.upsample3 = nn.Upsample(scale_factor=5, mode='nearest')             # [1,16,3600]
        self.dec_block3 = GroupConvBlock(16+16, 16, kernel_size=7, dilation=2)   # [1,16,3600] (16+16 due to skip connection)
        
        # Final output layer
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)                       # [1,1,3600]
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[LUNet] Model initialized with {total_params} parameters")
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the U-Net
        x: input tensor of shape (batch_size, input_dim) or (batch_size, 1, input_dim)
        returns: reconstructed signal of shape (batch_size, input_dim)
        """
        # Ensure input is float32
        x = x.float()
        
        # Reshape input if needed: (batch, input_dim) -> (batch, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Encoder path with skip connections storage
        # Level 1
        enc1 = self.enc_block1(x)           # [B, 16, 3600]
        x = self.pool1(enc1)                # [B, 16, 720]
        
        # Level 2  
        enc2 = self.enc_block2(x)           # [B, 32, 720]
        x = self.pool2(enc2)                # [B, 32, 360]
        
        # Level 3
        enc3 = self.enc_block3(x)           # [B, 64, 360]
        x = self.pool3(enc3)                # [B, 64, 180]
        
        # Bottleneck
        x = self.bottleneck(x)              # [B, 64, 180]
        
        # Decoder path with skip connections
        # Level 3 decode
        x = self.upsample1(x)               # [B, 64, 360]
        x = torch.cat([x, enc3], dim=1)     # [B, 128, 360] (64+64)
        x = self.dec_block1(x)              # [B, 32, 360]
        
        # Level 2 decode
        x = self.upsample2(x)               # [B, 32, 720]
        x = torch.cat([x, enc2], dim=1)     # [B, 64, 720] (32+32)
        x = self.dec_block2(x)              # [B, 16, 720]
        
        # Level 1 decode
        x = self.upsample3(x)               # [B, 16, 3600]
        x = torch.cat([x, enc1], dim=1)     # [B, 32, 3600] (16+16)
        x = self.dec_block3(x)              # [B, 16, 3600]
        
        # Final output
        x = self.final_conv(x)              # [B, 1, 3600]
        
        # Return as (batch, input_dim)
        x = x.squeeze(1)                    # [B, 3600]
        
        return x
    
    def loss(self, x, training=True):
        """
        Compute MSE loss for denoising
        x: input tensor (batch_size, input_dim)
        returns: dict with 'mse' loss
        """
        x = x.float()
        
        if x.dim() != 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
            
        x_recon = self.forward(x_flat)
        mse = F.mse_loss(x_recon.float(), x_flat.float(), reduction='mean')
        
        return {'mse': mse}


class AdaptiveLUNet(nn.Module):
    """
    Adaptive LUNet that can handle different input dimensions
    by adjusting the pooling/upsampling operations
    """
    def __init__(self, input_dim=370):  # Default to your actual signal length
        super().__init__()
        self.input_dim = input_dim
        
        print(f"[AdaptiveLUNet] Initializing Adaptive LUNet with input_dim={input_dim}")
        
        # Calculate pooling factors based on input dimension
        # We want to downsample to a reasonable bottleneck size
        self.pool_factors = self._calculate_pool_factors(input_dim)
        
        # Encoder
        self.enc_block1 = DeepSepConvBlock(1, 16, kernel_size=7, dilation=2)
        self.pool1 = nn.MaxPool1d(kernel_size=self.pool_factors[0], stride=self.pool_factors[0])
        
        self.enc_block2 = DeepSepConvBlock(16, 32, kernel_size=5, dilation=2)
        self.pool2 = nn.MaxPool1d(kernel_size=self.pool_factors[1], stride=self.pool_factors[1])
        
        self.enc_block3 = DeepSepConvBlock(32, 64, kernel_size=3, dilation=2)
        self.pool3 = nn.MaxPool1d(kernel_size=self.pool_factors[2], stride=self.pool_factors[2])
        
        # Bottleneck
        self.bottleneck = DeepSepConvBlock(64, 64, kernel_size=3, dilation=2)
        
        # Decoder
        self.dec_block1 = GroupConvBlock(64+64, 32, kernel_size=3, dilation=2)
        self.dec_block2 = GroupConvBlock(32+32, 16, kernel_size=5, dilation=2)
        self.dec_block3 = GroupConvBlock(16+16, 16, kernel_size=7, dilation=2)
        
        # Final output
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[AdaptiveLUNet] Model initialized with {total_params} parameters")
        print(f"[AdaptiveLUNet] Pool factors: {self.pool_factors}")
        
    def _calculate_pool_factors(self, input_dim):
        """Calculate appropriate pooling factors for the given input dimension"""
        # Target: reduce input_dim to roughly 20-50 in the bottleneck
        # Using 3 pooling operations
        
        if input_dim >= 1000:
            return [5, 2, 2]  # 1000 -> 200 -> 100 -> 50
        elif input_dim >= 500:
            return [3, 2, 2]  # 500 -> 166 -> 83 -> 41
        elif input_dim >= 200:
            return [2, 2, 2]  # 200 -> 100 -> 50 -> 25
        else:
            return [2, 2, 1]  # For smaller inputs, less aggressive pooling
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass with adaptive upsampling"""
        x = x.float()
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Store original length for final adjustment
        original_length = x.size(2)
        
        # Encoder with skip connections
        enc1 = self.enc_block1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc_block2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc_block3(x)
        x = self.pool3(enc3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections and adaptive upsampling
        # Upsample to match enc3 size
        x = F.interpolate(x, size=enc3.size(2), mode='nearest')
        x = torch.cat([x, enc3], dim=1)
        x = self.dec_block1(x)
        
        # Upsample to match enc2 size
        x = F.interpolate(x, size=enc2.size(2), mode='nearest')
        x = torch.cat([x, enc2], dim=1)
        x = self.dec_block2(x)
        
        # Upsample to match enc1 size
        x = F.interpolate(x, size=enc1.size(2), mode='nearest')
        x = torch.cat([x, enc1], dim=1)
        x = self.dec_block3(x)
        
        # Final conv
        x = self.final_conv(x)
        
        # Ensure output matches original input length exactly
        if x.size(2) != original_length:
            x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)
        
        return x.squeeze(1)
    
    def loss(self, x, training=True):
        """Compute MSE loss"""
        x = x.float()
        
        if x.dim() != 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
            
        x_recon = self.forward(x_flat)
        mse = F.mse_loss(x_recon.float(), x_flat.float(), reduction='mean')
        
        return {'mse': mse}


# For compatibility with your training script, create an alias
def LUNetModel(input_dim=370, adaptive=True):
    """
    Factory function to create the appropriate LUNet model
    
    Args:
        input_dim: Input signal dimension 
        adaptive: Whether to use adaptive version (recommended for non-3600 inputs)
    
    Returns:
        LUNet model instance
    """
    if adaptive or input_dim != 3600:
        return AdaptiveLUNet(input_dim=input_dim)
    else:
        return LUNet(input_dim=input_dim)