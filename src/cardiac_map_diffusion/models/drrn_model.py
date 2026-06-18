"""Deep Recurrent Residual Network (DRRN) baseline for cardiac MAP denoising.

This module defines an LSTM-based recurrent denoiser (``DRRN``), an enhanced
``AdaptiveDRRN`` variant (multi-layer LSTM with layer-norm and dropout), and a
``DRRNModel`` factory, following Antczak (2018), "Deep recurrent neural networks
for ECG signal denoising". This is a faithful, verbatim migration of the original
``drrn_model.py`` into the package layout; the LSTM/dense layer sizes,
activations, weight initialisation, and forward/loss math are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DRRN(nn.Module):
    """
    Deep Recurrent Neural Network for ECG signal denoising
    Based on: Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
    arXiv preprint arXiv:1807.11551.
    
    Original architecture:
    - LSTM(64, input_shape=(512, 1), return_sequences=True)
    - Dense(64, activation='relu')
    - Dense(64, activation='relu') 
    - Dense(1, activation='linear')
    """
    
    def __init__(self, input_dim=370, hidden_size=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        print(f"[DRRN] Initializing DRRN with input_dim={input_dim}, hidden_size={hidden_size}")
        
        # LSTM layer with return_sequences=True (equivalent to batch_first=True in PyTorch)
        self.lstm = nn.LSTM(
            input_size=1,           # Each timestep has 1 feature
            hidden_size=hidden_size, # 64 hidden units
            num_layers=1,
            batch_first=True,       # (batch, seq, feature) format
            dropout=0.0
        )
        
        # Dense layers (applied to each timestep)
        self.dense1 = nn.Linear(hidden_size, 64)
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[DRRN] Model initialized with {total_params} parameters")
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Only initialize 2D+ weight matrices
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            # Skip 1D parameters that aren't biases (LSTM internal params)
    
    def forward(self, x):
        """
        Forward pass through DRRN
        x: input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, 1)
        returns: denoised signal of shape (batch_size, input_dim)
        """
        # Ensure input is float32
        x = x.float()
        
        # Reshape input: (batch, input_dim) -> (batch, input_dim, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension
        
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        # lstm_out shape: (batch, seq_len, hidden_size)
        # h_n, c_n shapes: (1, batch, hidden_size) 
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply dense layers to each timestep
        # lstm_out: (batch, seq_len, hidden_size)
        dense1_out = self.relu(self.dense1(lstm_out))  # (batch, seq_len, 64)
        dense2_out = self.relu(self.dense2(dense1_out))  # (batch, seq_len, 64)
        output = self.dense3(dense2_out)  # (batch, seq_len, 1)
        
        # Remove the last dimension: (batch, seq_len, 1) -> (batch, seq_len)
        output = output.squeeze(-1)
        
        return output
    
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


class AdaptiveDRRN(nn.Module):
    """
    Adaptive DRRN that can handle different input dimensions
    while maintaining the core LSTM + Dense architecture
    """
    
    def __init__(self, input_dim=370, hidden_size=64, num_lstm_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        print(f"[AdaptiveDRRN] Initializing Adaptive DRRN with input_dim={input_dim}, hidden_size={hidden_size}")
        
        # Multi-layer LSTM (optional enhancement)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0.0,
            bidirectional=False  # Keep unidirectional like original
        )
        
        # Dense layers with potential residual connections
        self.dense1 = nn.Linear(hidden_size, 64)
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, 1)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[AdaptiveDRRN] Model initialized with {total_params} parameters")
        
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Only initialize 2D+ weight matrices
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            # Skip 1D parameters that aren't biases (LSTM internal params)
    
    def forward(self, x):
        """Forward pass with optional enhancements"""
        x = x.float()
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply layer normalization for stability
        lstm_out = self.layer_norm(lstm_out)
        
        # Dense layers with dropout
        dense1_out = self.dropout(self.relu(self.dense1(lstm_out)))
        dense2_out = self.dropout(self.relu(self.dense2(dense1_out)))
        output = self.dense3(dense2_out)
        
        return output.squeeze(-1)
    
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


# Factory function for easy model creation
def DRRNModel(input_dim=370, hidden_size=64, adaptive=True, num_lstm_layers=1):
    """
    Factory function to create the appropriate DRRN model
    
    Args:
        input_dim: Input signal dimension
        hidden_size: LSTM hidden size (default 64 as per paper)
        adaptive: Whether to use adaptive version with enhancements
        num_lstm_layers: Number of LSTM layers (1 for original, >1 for deeper)
    
    Returns:
        DRRN model instance
    """
    if adaptive:
        return AdaptiveDRRN(input_dim=input_dim, hidden_size=hidden_size, num_lstm_layers=num_lstm_layers)
    else:
        return DRRN(input_dim=input_dim, hidden_size=hidden_size)