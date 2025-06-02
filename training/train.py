import torch
import torch.optim as optim
from data.loaders.trajectory_loader import prepare_dataloaders
from models.safepath_model import SafePathModel
from training.losses.uncertainty_aware_loss import NegativeLogLikelihoodLoss

def train_safepath_model(config):
    """
    Train the SafePathAI model with uncertainty-aware training.
    
    Args:
        config: Training configuration
    
    Returns:
        Trained model
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Prepare data loaders
    train_loader, val_loader = prepare_dataloaders(
        batch_size=config.batch_size,
        num_workers=4
    )
    
    # Initialize model
    model = SafePathModel(config).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate
    )
    
    # Loss function
    criterion = NegativeLogLikelihoodLoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            mean, variance = model(inputs)
            
            # Compute loss
            loss = criterion(mean, variance, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                mean, variance = model(inputs)
                
                # Compute loss
                loss = criterion(mean, variance, targets)
                val_loss += loss.item()
        
        # Print epoch stats
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss}")
        