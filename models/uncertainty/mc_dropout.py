import torch
import torch.nn as nn

class MCDropout(nn.Module):
    """
    Monte Carlo Dropout module for epistemic uncertainty estimation.
    """
    
    def __init__(self, p=0.2):
        """
        Initialize MC Dropout.
        
        Args:
            p: Dropout probability
        """
        super(MCDropout, self).__init__()
        self.dropout = nn.Dropout(p=p)
    
    def forward(self, x):
        """
        Forward pass with dropout enabled.
        
        Args:
            x: Input tensor
        
        Returns:
            Dropout-applied tensor
        """
        return self.dropout(x)