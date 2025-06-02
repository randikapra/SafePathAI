import torch
import torch.nn as nn

class NegativeLogLikelihoodLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for probabilistic trajectory prediction.
    
    This loss function explicitly accounts for predicted variance,
    providing a principled way to learn both mean and variance.
    """
    
    def __init__(self, epsilon=1e-6):
        """
        Initialize the NLL Loss.
        
        Args:
            epsilon: Small constant to prevent numerical instability
        """
        super(NegativeLogLikelihoodLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, mean, variance, target):
        """
        Compute the Negative Log-Likelihood Loss.
        
        Args:
            mean: Predicted trajectory mean
            variance: Predicted trajectory variance
            target: Ground truth trajectory
        
        Returns:
            Computed loss value
        """
        # Ensure variance is positive
        variance = torch.clamp(variance, min=self.epsilon)
        
        # Compute squared error
        squared_error = torch.pow(target - mean, 2)
        
        # NLL Loss components
        log_variance = torch.log(variance)
        nll_loss = 0.5 * (log_variance + squared_error / variance)
        
        return torch.mean(nll_loss)

class CalibrationLoss(nn.Module):
    """
    Loss function to improve uncertainty calibration.
    
    This loss helps ensure that predicted uncertainties 
    accurately reflect the true prediction errors.
    """
    
    def __init__(self):
        super(CalibrationLoss, self).__init__()
    
    def forward(self, mean, variance, target):
        """
        Compute calibration loss.
        
        Args:
            mean: Predicted trajectory mean
            variance: Predicted trajectory variance
            target: Ground truth trajectory
        
        Returns:
            Calibration loss value
        """
        # Compute prediction error
        error = torch.abs(target - mean)
        
        # Normalized variance 
        normalized_variance = variance / torch.max(variance)
        
        # Calibration loss tries to align error with predicted variance
        calibration_loss = torch.mean(torch.abs(error - normalized_variance))
        
        return calibration_loss