import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from models.safepath_model import HybridTrajectoryPredictor
from data.loaders.multi_dataset_loader import prepare_multi_modal_dataloaders
from configs.config_manager import Config
    

def plot_trajectory_uncertainty(history, 
                                true_future, 
                                predicted_future, 
                                uncertainty,
                                title='Trajectory Prediction with Uncertainty'):
    """
    Comprehensive visualization of trajectory prediction and uncertainty
    
    Args:
        history: Historical trajectory [seq_len, 2]
        true_future: Ground truth future trajectory [pred_len, 2]
        predicted_future: Predicted future trajectory [pred_len, 2]
        uncertainty: Prediction uncertainty [pred_len, 2]
        title: Plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Trajectory Plot
    plt.subplot(2, 2, 1)
    plt.title('Trajectory Comparison')
    plt.plot(history[:, 0], history[:, 1], 'k-', label='History')
    plt.plot(true_future[:, 0], true_future[:, 1], 'g-', label='Ground Truth')
    plt.plot(predicted_future[:, 0], predicted_future[:, 1], 'r-', label='Prediction')
    plt.grid(True)
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Uncertainty Heatmap
    plt.subplot(2, 2, 2)
    plt.title('Prediction Uncertainty')
    sns.heatmap(uncertainty, cmap='YlOrRd', annot=True, fmt='.4f', 
                cbar_kws={'label': 'Uncertainty'})
    plt.xlabel('Coordinate')
    plt.ylabel('Prediction Step')
    
    # Uncertainty Progression
    plt.subplot(2, 2, 3)
    plt.title('Uncertainty Progression')
    uncertainty_mean = np.mean(uncertainty, axis=1)
    plt.plot(uncertainty_mean, marker='o')
    plt.fill_between(range(len(uncertainty_mean)), 
                     uncertainty_mean - np.std(uncertainty, axis=1),
                     uncertainty_mean + np.std(uncertainty, axis=1), 
                     alpha=0.3)
    plt.xlabel('Prediction Step')
    plt.ylabel('Mean Uncertainty')
    
    # Position Error
    plt.subplot(2, 2, 4)
    plt.title('Position Error')
    position_error = np.abs(predicted_future - true_future)
    plt.plot(position_error[:, 0], label='X Error')
    plt.plot(position_error[:, 1], label='Y Error')
    plt.legend()
    plt.xlabel('Prediction Step')
    plt.ylabel('Absolute Error')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.show()

def analyze_model_performance(model, test_loader, device):
    """
    Perform comprehensive model performance and uncertainty analysis
    
    Args:
        model: Trained hybrid trajectory predictor
        test_loader: DataLoader for test data
        device: Torch device
    """
    model.eval()
    total_uncertainties = []
    total_errors = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Predictions with uncertainty
            predictions, uncertainties, confidence_scores = model.predict(inputs)
            
            # Convert to numpy for analysis
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()
            uncertainties_np = uncertainties.cpu().numpy()
            
            # Calculate errors
            errors = np.abs(predictions_np - targets_np)
            
            total_uncertainties.append(uncertainties_np)
            total_errors.append(errors)
    
    # Aggregate results
    total_uncertainties = np.concatenate(total_uncertainties, axis=0)
    total_errors = np.concatenate(total_errors, axis=0)
    
    # Uncertainty-Error Correlation
    correlation_x = np.corrcoef(total_uncertainties[:, :, 0].flatten(), 
                                 total_errors[:, :, 0].flatten())[0, 1]
    correlation_y = np.corrcoef(total_uncertainties[:, :, 1].flatten(), 
                                 total_errors[:, :, 1].flatten())[0, 1]
    
    print(f"Uncertainty-Error Correlation (X): {correlation_x:.4f}")
    print(f"Uncertainty-Error Correlation (Y): {correlation_y:.4f}")
    
    # Visualize a sample trajectory
    plot_trajectory_uncertainty(
        history=inputs[0, :, :2].cpu().numpy(),
        true_future=targets[0].cpu().numpy(),
        predicted_future=predictions[0, :, :2].cpu().numpy(),
        uncertainty=uncertainties[0, :, :2].cpu().numpy()
    )

def main():
    """
    Example usage of trajectory uncertainty visualization
    """
    # Configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    _, test_loader = prepare_multi_modal_dataloaders(config)
    
    # Create and load trained model
    hybrid_model = HybridTrajectoryPredictor(config, device)
    hybrid_model.load_checkpoint()  # Implement checkpoint loading
    
    # Analyze model performance
    analyze_model_performance(hybrid_model, test_loader, device)

if __name__ == "__main__":
    main()