"""
SafePathAI: Uncertainty-Aware Trajectory Prediction for Safe Autonomous Driving
Main implementation code
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union, Callable


#######################
# CONFIG
#######################

class Config:
    """Configuration for the SafePathAI model."""
    # Data parameters
    input_seq_len = 20  # Length of input trajectory sequence
    pred_seq_len = 30   # Length of predicted trajectory sequence
    input_dim = 4       # (x, y, vx, vy)
    output_dim = 2      # (x, y)
    
    # Model parameters
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    mc_dropout_samples = 20
    ensemble_size = 5
    
    # Training parameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100
    
    # Kalman Filter parameters
    q_var = 0.01  # Process noise variance
    r_var = 0.1   # Measurement noise variance
    
    # Uncertainty thresholds
    uncertainty_threshold = 0.5  # Threshold for high uncertainty


#######################
# DATA PROCESSING
#######################

class TrajectoryDataset(Dataset):
    """Dataset for trajectory prediction."""
    
    def __init__(self, data_path: str, seq_len: int, pred_len: int, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset
            seq_len: Length of input sequence
            pred_len: Length of prediction sequence
            transform: Optional transform to apply to the data
        """
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transform = transform
        
        # Load the data (simplified for demonstration)
        # In a real implementation, use actual dataset loaders
        self.data = self._load_data()
        
    def _load_data(self):
        """Load and preprocess the trajectory data."""
        # This is a placeholder - in a real implementation, 
        # load actual data from nuScenes, Argoverse, etc.
        # Simulating data for demonstration
        num_samples = 1000
        total_seq_length = self.seq_len + self.pred_len
        
        # Generate synthetic trajectories (X, Y, Vx, Vy)
        trajectories = []
        for _ in range(num_samples):
            # Generate random starting position and velocity
            x0, y0 = np.random.uniform(-10, 10, 2)
            vx0, vy0 = np.random.uniform(-2, 2, 2)
            
            # Generate trajectory with some noise
            trajectory = np.zeros((total_seq_length, 4))
            trajectory[0] = [x0, y0, vx0, vy0]
            
            for t in range(1, total_seq_length):
                # Add some random acceleration
                ax, ay = np.random.normal(0, 0.1, 2)
                vx = trajectory[t-1, 2] + ax
                vy = trajectory[t-1, 3] + ay
                x = trajectory[t-1, 0] + vx
                y = trajectory[t-1, 1] + vy
                trajectory[t] = [x, y, vx, vy]
            
            trajectories.append(trajectory)
        
        return trajectories
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        trajectory = self.data[idx]
        
        # Split into input and target sequences
        input_seq = trajectory[:self.seq_len]
        target_seq = trajectory[self.seq_len:, :2]  # Only position (x,y) for target
        
        # Convert to tensors
        input_seq = torch.FloatTensor(input_seq)
        target_seq = torch.FloatTensor(target_seq)
        
        if self.transform:
            input_seq = self.transform(input_seq)
            
        return input_seq, target_seq


def prepare_dataloaders(config: Config):
    """Prepare data loaders for training and validation."""
    # In a real implementation, you would load actual datasets
    dataset = TrajectoryDataset(
        data_path="./data",
        seq_len=config.input_seq_len,
        pred_len=config.pred_seq_len
    )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader


#######################
# KALMAN FILTER
#######################

class KalmanFilter:
    """
    Kalman Filter for trajectory prediction with uncertainty estimation.
    """
    
    def __init__(self, q_var: float = 0.01, r_var: float = 0.1):
        """
        Initialize the Kalman Filter.
        
        Args:
            q_var: Process noise variance
            r_var: Measurement noise variance
        """
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (we observe x and y positions)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * q_var
        
        # Measurement noise covariance
        self.R = np.eye(2) * r_var
        
        # Initial state and covariance
        self.x = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4)
        
    def predict(self):
        """
        Predict the next state using the Kalman filter.
        
        Returns:
            predicted_state: The predicted state [x, y, vx, vy]
            uncertainty: The uncertainty of the prediction (covariance)
        """
        # Predict the state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Return the predicted position and its uncertainty
        predicted_position = self.x[:2]
        position_uncertainty = self.P[:2, :2]
        
        return predicted_position, position_uncertainty
    
    def update(self, measurement: np.ndarray):
        """
        Update the state estimate with a new measurement.
        
        Args:
            measurement: The observed position [x, y]
        """
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        y = measurement - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def reset(self, initial_state: np.ndarray):
        """
        Reset the filter with a new initial state.
        
        Args:
            initial_state: The initial state [x, y, vx, vy]
        """
        self.x = initial_state
        self.P = np.eye(4)
    
    def get_state_with_uncertainty(self):
        """
        Get the current state and its uncertainty.
        
        Returns:
            state: The current state [x, y, vx, vy]
            uncertainty: The uncertainty of the state (covariance)
        """
        return self.x, self.P


#######################
# LSTM MODEL
#######################

class LSTMTrajectoryPredictor(nn.Module):
    """
    LSTM-based trajectory predictor with attention and uncertainty estimation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the LSTM trajectory predictor.
        
        Args:
            config: Model configuration
        """
        super(LSTMTrajectoryPredictor, self).__init__()
        self.config = config
        
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )
        
        # Decoder (predicts mean and variance)
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Output layers for mean and variance (aleatoric uncertainty)
        self.mean_predictor = nn.Linear(config.hidden_size, config.pred_seq_len * config.output_dim)
        self.var_predictor = nn.Linear(config.hidden_size, config.pred_seq_len * config.output_dim)
        
    def forward(self, x: torch.Tensor, mc_dropout: bool = False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mc_dropout: Whether to use MC dropout for epistemic uncertainty
        
        Returns:
            mean: Predicted trajectory mean
            var: Predicted trajectory variance (aleatoric uncertainty)
        """
        batch_size = x.size(0)
        
        # Set dropout mode
        if mc_dropout:
            self.train()  # Use dropout during inference
        
        # Encode the input sequence
        lstm_out, _ = self.lstm_encoder(x)  # [batch_size, seq_len, hidden_size]
        
        # Apply attention
        attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_size]
        
        # Decode
        features = self.decoder(context_vector)
        
        # Predict mean and variance
        mean = self.mean_predictor(features)
        log_var = self.var_predictor(features)  # Log variance for numerical stability
        
        # Reshape to [batch_size, pred_seq_len, output_dim]
        mean = mean.view(batch_size, self.config.pred_seq_len, self.config.output_dim)
        log_var = log_var.view(batch_size, self.config.pred_seq_len, self.config.output_dim)
        var = torch.exp(log_var)
        
        return mean, var
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 20):
        """
        Predict trajectory with both aleatoric and epistemic uncertainty.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            num_samples: Number of MC dropout samples
        
        Returns:
            mean: Mean predicted trajectory
            aleatoric_uncertainty: Aleatoric uncertainty (from output variance)
            epistemic_uncertainty: Epistemic uncertainty (from MC dropout)
            total_uncertainty: Combined uncertainty
        """
        batch_size = x.size(0)
        pred_len = self.config.pred_seq_len
        output_dim = self.config.output_dim
        
        # Storage for MC dropout samples
        all_means = torch.zeros(num_samples, batch_size, pred_len, output_dim).to(x.device)
        all_vars = torch.zeros(num_samples, batch_size, pred_len, output_dim).to(x.device)
        
        # Run multiple forward passes with dropout
        for i in range(num_samples):
            mean, var = self.forward(x, mc_dropout=True)
            all_means[i] = mean
            all_vars[i] = var
        
        # Calculate predictive mean and uncertainties
        pred_mean = all_means.mean(dim=0)  # [batch_size, pred_len, output_dim]
        
        # Aleatoric uncertainty (average of predicted variances)
        aleatoric_uncertainty = all_vars.mean(dim=0)  # [batch_size, pred_len, output_dim]
        
        # Epistemic uncertainty (variance of predicted means)
        epistemic_uncertainty = all_means.var(dim=0)  # [batch_size, pred_len, output_dim]
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return pred_mean, aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty


#######################
# ENSEMBLE MODEL
#######################

class EnsembleModel:
    """
    Ensemble of trajectory prediction models for improved uncertainty estimation.
    """
    
    def __init__(self, config: Config, device: torch.device):
        """
        Initialize the ensemble model.
        
        Args:
            config: Model configuration
            device: Device to run the models on
        """
        self.config = config
        self.device = device
        self.models = []
        
        # Create ensemble of models
        for _ in range(config.ensemble_size):
            model = LSTMTrajectoryPredictor(config).to(device)
            self.models.append(model)
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, model_idx: int):
        """
        Train a single model in the ensemble.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_idx: Index of the model to train
        """
        model = self.models[model_idx]
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # NLL loss for probabilistic prediction
        def nll_loss(mean, var, target):
            """Negative log-likelihood loss for Gaussian distribution."""
            return torch.mean(0.5 * torch.log(var) + 0.5 * (target - mean)**2 / var)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                mean, var = model(inputs)
                loss = nll_loss(mean, var + 1e-6, targets)  # Add small epsilon for numerical stability
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    mean, var = model(inputs)
                    loss = nll_loss(mean, var + 1e-6, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_{model_idx}.pt")
            
            print(f"Model {model_idx}, Epoch {epoch+1}/{self.config.num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def train_all_models(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train all models in the ensemble.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        for i in range(len(self.models)):
            print(f"Training model {i+1}/{len(self.models)}")
            self.train_model(train_loader, val_loader, i)
    
    def predict_with_uncertainty(self, x: torch.Tensor):
        """
        Make predictions with the ensemble and estimate uncertainty.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        
        Returns:
            ensemble_mean: Mean prediction across all models
            aleatoric_uncertainty: Uncertainty from model outputs
            epistemic_uncertainty: Uncertainty from model disagreement
            total_uncertainty: Combined uncertainty
        """
        batch_size = x.size(0)
        pred_len = self.config.pred_seq_len
        output_dim = self.config.output_dim
        num_models = len(self.models)
        
        all_means = torch.zeros(num_models, batch_size, pred_len, output_dim).to(self.device)
        all_vars = torch.zeros(num_models, batch_size, pred_len, output_dim).to(self.device)
        
        # Get predictions from all models
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                mean, var = model(x)
                all_means[i] = mean
                all_vars[i] = var
        
        # Ensemble mean prediction
        ensemble_mean = all_means.mean(dim=0)
        
        # Aleatoric uncertainty (average of model variances)
        aleatoric_uncertainty = all_vars.mean(dim=0)
        
        # Epistemic uncertainty (variance of model means)
        epistemic_uncertainty = all_means.var(dim=0)
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return ensemble_mean, aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty


#######################
# HYBRID MODEL
#######################

class HybridTrajectoryPredictor:
    """
    Hybrid trajectory predictor combining Kalman Filter and deep learning models.
    """
    
    def __init__(self, config: Config, device: torch.device):
        """
        Initialize the hybrid predictor.
        
        Args:
            config: Model configuration
            device: Device to run the models on
        """
        self.config = config
        self.device = device
        
        # Kalman filter
        self.kalman_filter = KalmanFilter(q_var=config.q_var, r_var=config.r_var)
        
        # Deep learning ensemble
        self.ensemble_model = EnsembleModel(config, device)
        
        # Confidence weights (initialized equally)
        self.kalman_weight = 0.5
        self.dl_weight = 0.5
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the deep learning components of the hybrid model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.ensemble_model.train_all_models(train_loader, val_loader)
    
    def predict(self, trajectory_history: torch.Tensor):
        """
        Make predictions using the hybrid model.
        
        Args:
            trajectory_history: Input tensor of shape [batch_size, seq_len, input_dim]
        
        Returns:
            hybrid_prediction: Combined prediction
            uncertainty: Prediction uncertainty
            confidence: Model confidence (inverse of uncertainty)
        """
        batch_size = trajectory_history.size(0)
        
        # Get the last observed state for Kalman Filter initialization
        last_observed = trajectory_history[:, -1, :].detach().cpu().numpy()
        
        # Kalman Filter predictions (for each sequence in the batch)
        kf_predictions = np.zeros((batch_size, self.config.pred_seq_len, 2))
        kf_uncertainties = np.zeros((batch_size, self.config.pred_seq_len, 2, 2))
        
        for b in range(batch_size):
            # Reset Kalman filter with last observed state
            self.kalman_filter.reset(last_observed[b])
            
            # Make predictions
            for t in range(self.config.pred_seq_len):
                pos, uncertainty = self.kalman_filter.predict()
                kf_predictions[b, t] = pos
                kf_uncertainties[b, t] = uncertainty
        
        # Convert to torch tensors
        kf_predictions = torch.FloatTensor(kf_predictions).to(self.device)
        kf_uncertainties_diag = torch.FloatTensor(np.array([
            [kf_uncertainties[b, t, 0, 0] + kf_uncertainties[b, t, 1, 1] for t in range(self.config.pred_seq_len)]
            for b in range(batch_size)
        ])).to(self.device).unsqueeze(-1).repeat(1, 1, 2)
        
        # Deep learning predictions
        dl_predictions, aleatoric_uncertainty, epistemic_uncertainty, dl_total_uncertainty = \
            self.ensemble_model.predict_with_uncertainty(trajectory_history)
        
        # Dynamic weighting based on uncertainties
        # Lower uncertainty -> higher weight
        kf_confidence = 1.0 / (kf_uncertainties_diag + 1e-6)
        dl_confidence = 1.0 / (dl_total_uncertainty + 1e-6)
        
        total_confidence = kf_confidence + dl_confidence
        kf_weight = kf_confidence / total_confidence
        dl_weight = dl_confidence / total_confidence
        
        # Weighted combination of predictions
        hybrid_prediction = kf_weight * kf_predictions + dl_weight * dl_predictions
        
        # Combined uncertainty (weighted average)
        combined_uncertainty = (kf_weight**2 * kf_uncertainties_diag + 
                               dl_weight**2 * dl_total_uncertainty)
        
        # Overall confidence score (inverse of uncertainty)
        confidence_score = 1.0 / (torch.mean(combined_uncertainty, dim=(1, 2)) + 1e-6)
        
        return hybrid_prediction, combined_uncertainty, confidence_score
    
    def predict_with_rejection(self, trajectory_history: torch.Tensor):
        """
        Make predictions with a rejection option for high uncertainty cases.
        
        Args:
            trajectory_history: Input tensor of shape [batch_size, seq_len, input_dim]
        
        Returns:
            predictions: Trajectory predictions
            uncertainties: Prediction uncertainties
            valid_mask: Boolean mask indicating which predictions are valid
        """
        predictions, uncertainties, confidence_scores = self.predict(trajectory_history)
        
        # Create a mask for predictions with low confidence
        valid_mask = confidence_scores > (1.0 / self.config.uncertainty_threshold)
        
        return predictions, uncertainties, valid_mask


#######################
# VISUALIZATION
#######################

def visualize_prediction(history: np.ndarray, 
                        true_future: np.ndarray, 
                        predicted_future: np.ndarray,
                        uncertainty: np.ndarray = None):
    """
    Visualize trajectory history, ground truth future, and prediction with uncertainty.
    
    Args:
        history: Historical trajectory [seq_len, 2]
        true_future: Ground truth future trajectory [pred_len, 2]
        predicted_future: Predicted future trajectory [pred_len, 2]
        uncertainty: Prediction uncertainty [pred_len, 2]
    """
    plt.figure(figsize=(10, 8))
    
    # Plot history
    plt.plot(history[:, 0], history[:, 1], 'k-', label='History')
    plt.plot(history[-1, 0], history[-1, 1], 'ko', markersize=8)
    
    # Plot ground truth future
    plt.plot(true_future[:, 0], true_future[:, 1], 'g-', label='Ground Truth')
    
    # Plot prediction
    plt.plot(predicted_future[:, 0], predicted_future[:, 1], 'r-', label='Prediction')
    
    # Plot uncertainty ellipses if provided
    if uncertainty is not None:
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        
        for i in range(0, len(predicted_future), 5):  # Plot every 5th step for clarity
            position = predicted_future[i]
            covariance = uncertainty[i]
            
            # Create confidence ellipse
            eigvals, eigvecs = np.linalg.eigh(covariance)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2 * np.sqrt(5.991 * eigvals)  # 95% confidence
            
            ellipse = Ellipse(xy=position, width=width, height=height, angle=angle,
                             alpha=0.3, color='red')
            plt.gca().add_patch(ellipse)
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory Prediction with Uncertainty')
    plt.axis('equal')
    plt.show()


#######################
# MAIN
#######################

def main():
    """Main function to demonstrate the SafePathAI model."""
    # Configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader = prepare_dataloaders(config)
    
    # Create hybrid model
    hybrid_model = HybridTrajectoryPredictor(config, device)
    
    # Train the model
    print("Training the model...")
    hybrid_model.train(train_loader, val_loader)
    print("Training complete!")
    
    # Evaluate on a sample
    print("Evaluating on a sample...")
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Make predictions
        predictions, uncertainties, confidence_scores = hybrid_model.predict(inputs)
        
        # Get one sample for visualization
        sample_idx = 0
        sample_input = inputs[sample_idx].detach().cpu().numpy()
        sample_target = targets[sample_idx].detach().cpu().numpy()
        sample_pred = predictions[sample_idx].detach().cpu().numpy()
        sample_uncertainty = uncertainties[sample_idx].detach().cpu().numpy()
        
        # Create uncertainty matrix for visualization
        uncertainty_matrices = np.zeros((config.pred_seq_len, 2, 2))
        for t in range(config.pred_seq_len):
            # Diagonal uncertainty
            uncertainty_matrices[t, 0, 0] = sample_uncertainty[t, 0]
            uncertainty_matrices[t, 1, 1] = sample_uncertainty[t, 1]
        
        # Visualize
        visualize_prediction(
            history=sample_input[:, :2],
            true_future=sample_target,
            predicted_future=sample_pred,
            uncertainty=uncertainty_matrices
        )
        
        break  # Just one batch for demo
    
    print("Demo complete!")


if __name__ == "__main__":
    main()
