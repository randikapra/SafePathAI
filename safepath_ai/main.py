import argparse
import torch

from configs.config_manager import ConfigManager
from training.train import train_safepath_model
from inference.predict import SafePathPredictor
from evaluation.metrics.trajectory_metrics import evaluate_trajectory_prediction

def main():
    """
    Main entry point for SafePathAI.
    
    Handles training, evaluation, and inference for the trajectory prediction model.
    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="SafePathAI: Uncertainty-Aware Trajectory Prediction")
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'predict', 'evaluate'],
                        help='Operation mode: train, predict, or evaluate')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Input trajectory file for prediction/evaluation')
    parser.add_argument('--output', type=str, help='Output file for predictions')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager.load_config(args.config)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mode-specific operations
    if args.mode == 'train':
        print("Training SafePathAI model...")
        trained_model = train_safepath_model(config)
        
        # Save the trained model
        torch.save(trained_model.state_dict(), 'safepath_model.pth')
        print("Model training completed and saved.")
    
    elif args.mode == 'predict':
        print("Running trajectory prediction...")
        
        # Load trained model
        model = torch.load('safepath_model.pth')
        predictor = SafePathPredictor(model, device)
        
        # Load input trajectory
        if not args.input:
            raise ValueError("Input trajectory file is required for prediction")
        
        # Perform prediction
        predictions, uncertainties = predictor.predict(args.input)
        
        # Save predictions
        output_path = args.output or 'predictions.json'
        predictor.save_predictions(predictions, uncertainties, output_path)
        
        print(f"Predictions saved to {output_path}")
    
    elif args.mode == 'evaluate':
        print("Evaluating model performance...")
        
        # Load trained model
        model = torch.load('safepath_model.pth')
        
        # Perform evaluation
        metrics = evaluate_trajectory_prediction(
            model, 
            config, 
            input_file=args.input
        )
        
        # Print and potentially save metrics
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()