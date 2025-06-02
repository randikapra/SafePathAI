import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory prediction with synthetic data generation.
    """
    
    def __init__(self, num_samples=1000, seq_len=20, pred_len=30):
        """
        Initialize the trajectory dataset.
        
        Args:
            num_samples: Number of synthetic trajectories
            seq_len: Length of input sequence
            pred_len: Length of prediction sequence
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.trajectories = self._generate_synthetic_trajectories()
    
    def _generate_synthetic_trajectories(self):
        """
        Generate synthetic trajectory data.
        
        Returns:
            List of trajectory sequences
        """
        trajectories = []
        total_seq_length = self.seq_len + self.pred_len
        
        for _ in range(self.num_samples):
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
        return self.num_samples
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        
        # Split into input and target sequences
        input_seq = trajectory[:self.seq_len]
        target_seq = trajectory[self.seq_len:, :2]  # Only position (x,y) for target
        
        # Convert to tensors
        input_seq = torch.FloatTensor(input_seq)
        target_seq = torch.FloatTensor(target_seq)
        
        return input_seq, target_seq

def prepare_dataloaders(batch_size=64, num_workers=4):
    """
    Prepare data loaders for training and validation.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
    
    Returns:
        Train and validation data loaders
    """
    # Create full dataset
    dataset = TrajectoryDataset()
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader