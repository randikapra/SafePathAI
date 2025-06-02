import os
import yaml
from typing import Dict, Any
import torch

class ConfigManager:
    """
    Advanced configuration management for SafePathAI.
    Supports YAML-based configuration with environment and CLI overrides.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self._validate_config()
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with fallback to default values.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Loaded configuration dictionary
        """
        default_config = {
            'data': {
                'datasets': ['nuScenes', 'Argoverse'],
                'input_seq_len': 20,
                'pred_seq_len': 30,
                'augmentation': True
            },
            'model': {
                'type': 'hybrid',
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'ensemble_size': 5
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 0.001,
                'num_epochs': 100,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'uncertainty': {
                'mc_dropout_samples': 20,
                'threshold': 0.5,
                'quantification_method': 'ensemble'
            }
        }
        
        # Load from YAML if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Deep merge of default and user configurations
            self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict):
        """
        Recursively merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary to update base with
        """
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = self._deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base
    
    def _validate_config(self):
        """
        Validate configuration parameters.
        Perform sanity checks and set constraints.
        """
        assert self.config['training']['batch_size'] > 0, "Batch size must be positive"
        assert self.config['model']['hidden_size'] > 0, "Hidden size must be positive"
        assert 0 <= self.config['model']['dropout'] <= 1, "Dropout must be between 0 and 1"
        
        # Ensure device is valid
        valid_devices = ['cpu', 'cuda']
        assert self.config['training']['device'] in valid_devices, f"Invalid device: {self.config['training']['device']}"
    
    def get_config(self, section: str = None) -> Dict[str, Any]:
        """
        Retrieve configuration.
        
        Args:
            section: Optional section to retrieve
        
        Returns:
            Configuration dictionary or specific section
        """
        if section:
            return self.config.get(section, {})
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration dynamically.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self._deep_merge(self.config, updates)
        self._validate_config()
    
    def save_config(self, path: str):
        """
        Save current configuration to a YAML file.
        
        Args:
            path: File path to save configuration
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

@dataclass
class Config:
    """
    Comprehensive configuration for SafePathAI model
    """
    # Data parameters
    input_seq_len: int = 20
    pred_seq_len: int = 30
    input_dim: int = 4
    output_dim: int = 2
    
    # Multi-dataset configuration
    datasets: List[str] = field(default_factory=lambda: [
        'nuscenes', 'argoverse', 'dut', 'marinetraffic', 'crowdflow'
    ])
    
    # Model parameters
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    mc_dropout_samples: int = 20
    ensemble_size: int = 5
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 100
    
    # Kalman Filter parameters
    q_var: float = 0.01
    r_var: float = 0.1
    
    # Uncertainty thresholds
    uncertainty_threshold: float = 0.5
    
    # Paths
    base_data_dir: str = './data'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    
    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None):
        """
        Load configuration from YAML file
        
        Args:
            yaml_path: Path to configuration YAML
        
        Returns:
            Configured Config object
        """
        if yaml_path is None:
            # Default path
            yaml_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        # If file doesn't exist, create with default values
        if not os.path.exists(yaml_path):
            default_config = asdict(cls())
            with open(yaml_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        
        # Load configuration
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: Optional[str] = None):
        """
        Save configuration to YAML file
        
        Args:
            yaml_path: Path to save configuration YAML
        """
        if yaml_path is None:
            yaml_path = os.path.join(self.checkpoint_dir, 'last_config.yaml')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        # Save configuration
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def update(self, **kwargs):
        """
        Update configuration dynamically
        
        Args:
            **kwargs: Key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config does not have attribute {key}")
        
        return self
    
    def get_dataset_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate dataset-specific configurations
        
        Returns:
            Dictionary of dataset-specific configurations
        """
        dataset_configs = {}
        for dataset in self.datasets:
            dataset_configs[dataset] = {
                'seq_len': self.input_seq_len,
                'pred_len': self.pred_seq_len,
                'data_dir': os.path.join(self.base_data_dir, dataset)
            }
        return dataset_configs

import yaml
from dataclasses import dataclass, asdict

@dataclass
class SafePathConfig:
    """
    Comprehensive configuration for SafePathAI model.
    """
    # Data parameters
    input_seq_len: int = 20
    pred_seq_len: int = 30
    input_dim: int = 4
    output_dim: int = 2
    
    # Model parameters
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 100
    
    # Uncertainty parameters
    mc_dropout_samples: int = 20
    ensemble_size: int = 5
    
    # Kalman Filter parameters
    q_var: float = 0.01
    r_var: float = 0.1
    
    # Uncertainty thresholds
    uncertainty_threshold: float = 0.5

class ConfigManager:
    """
    Configuration management for SafePathAI.
    
    Supports loading, saving, and managing model configurations.
    """
    
    @staticmethod
    def load_config(config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        
        Returns:
            SafePathConfig object
        """
        try:
            with open(config_path, 'r') as file:
                config_dict = yaml.safe_load(file)
            return SafePathConfig(**config_dict)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. Using default configuration.")
            return SafePathConfig()
        except Exception as e:
            print(f"Error loading config: {e}")
            return SafePathConfig()
    
    @staticmethod
    def save_config(config, save_path):
        """
        Save configuration to a YAML file.
        
        Args:
            config: SafePathConfig object
            save_path: Path to save the configuration
        """
        try:
            with open(save_path, 'w') as file:
                yaml.dump(asdict(config), file, default_flow_style=False)
            print(f"Configuration saved to {save_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    @staticmethod
    def create_default_config():
        """
        Create and return a default configuration.
        
        Returns:
            SafePathConfig with default parameters
        """
        return SafePathConfig()

# Example configuration files
DEFAULT_CONFIG = """
# SafePathAI Default Configuration
input_seq_len: 20
pred_seq_len: 30
input_dim: 4
output_dim: 2
hidden_size: 128
num_layers: 2
dropout: 0.2
batch_size: 64
learning_rate: 0.001
num_epochs: 100
mc_dropout_samples: 20
ensemble_size: 5
uncertainty_threshold: 0.5
"""

# Create default config files if they don't exist
def _create_default_config_files():
    """
    Create default configuration files.
    """
    import os
    
    # Ensure configs directory exists
    os.makedirs('configs', exist_ok=True)
    
    # Create default config
    default_config_path = 'configs/default_config.yaml'
    if not os.path.exists(default_config_path):
        with open(default_config_path, 'w') as f:
            f.write(DEFAULT_CONFIG)
        print(f"Created default configuration at {default_config_path}")

# Run config file creation
_create_default_config_files()\

# Example usage
if __name__ == "__main__":
    config_manager = ConfigManager('config.yaml')
    print(config_manager.get_config('model'))
    
    # Dynamic update
    config_manager.update_config({
        'training': {
            'learning_rate': 0.0005
        }
    })
    
    # Save current configuration
    config_manager.save_config('updated_config.yaml')