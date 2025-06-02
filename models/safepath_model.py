import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .encoders.trajectory_encoder import TrajectoryEncoder
from .uncertainty.mc_dropout import MCDropout
from .uncertainty.deep_ensemble import DeepEnsemble
from .decoders.probabilistic_decoder import ProbabilisticDecoder

class SafePathModel(nn.Module):
    """
    SafePathAI: Uncertainty-Aware Trajectory Prediction Model
    
    This model combines advanced trajectory prediction techniques with 
    comprehensive uncertainty estimation.
    """
    
    def __init__(self, config):
        """
        Initialize the SafePathAI model.
        
        Args:
            config: Configuration object with model hyperparameters
        """
        super(SafePathModel, self).__init__()
        
        # Trajectory Encoder
        self.encoder = TrajectoryEncoder(
            input_dim=config.input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers
        )
        
        # MC Dropout for Epistemic Uncertainty
        self.mc_dropout = MCDropout(p=config.dropout)
        
        # Probabilistic Decoder
        self.decoder = ProbabilisticDecoder(
            input_size=config.hidden_size,
            output_dim=config.output_dim,
            pred_seq_len=config.pred_seq_len
        )
    
    def forward(self, x, mc_dropout=False):
        """
        Forward pass of the SafePathAI model.
        
        Args:
            x: Input trajectory tensor
            mc_dropout: Flag to enable Monte Carlo Dropout
        
        Returns:
            Predicted trajectory mean and variance
        """
        # Encode input trajectory
        encoded_features = self.encoder(x)
        
        # Apply MC Dropout if enabled
        if mc_dropout:
            encoded_features = self.mc_dropout(encoded_features)
        
        # Decode to get probabilistic trajectory prediction
        mean, variance = self.decoder(encoded_features)
        
        return mean, variance

    def predict_with_uncertainty(self, x, num_samples=20):
        """
        Predict trajectory with comprehensive uncertainty estimation.
        
        Args:
            x: Input trajectory tensor
            num_samples: Number of MC Dropout samples
        
        Returns:
            Predictive mean, aleatoric, epistemic, and total uncertainty
        """
        # Storage for MC dropout samples
        all_means = torch.zeros(num_samples, *x.shape[:-1], 2)
        all_vars = torch.zeros_like(all_means)
        
        # Multiple forward passes with dropout
        for i in range(num_samples):
            mean, var = self.forward(x, mc_dropout=True)
            all_means[i] = mean
            all_vars[i] = var
        
        # Compute uncertainties
        pred_mean = all_means.mean(dim=0)
        aleatoric_uncertainty = all_vars.mean(dim=0)
        epistemic_uncertainty = all_means.var(dim=0)
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return pred_mean, aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty