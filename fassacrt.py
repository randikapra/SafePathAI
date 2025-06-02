import os

# Define the project structure
project_structure = {
    "safepath_ai": [
        "main.py", "train_model.py", "evaluate_model.py", "predict.py",
        "requirements.txt", "setup.py", "Dockerfile", "README.md"
    ],
    "data": ["dataset.py"],
    "data/loaders": ["nuscenes_loader.py", "argoverse_loader.py", "interaction_loader.py"],
    "data/preprocessing": ["trajectory_preprocessing.py", "map_preprocessing.py", "feature_extraction.py"],
    "data/augmentation": ["noise_injection.py", "trajectory_augmentation.py", "synthetic_scenarios.py"],
    "models": ["safepath_model.py"],
    "models/encoders": ["trajectory_encoder.py", "map_encoder.py", "interaction_encoder.py"],
    "models/uncertainty": ["bayesian_layers.py", "mc_dropout.py", "deep_ensemble.py", "evidential_learning.py"],
    "models/decoders": ["trajectory_decoder.py", "multimodal_decoder.py", "probabilistic_decoder.py"],
    "models/fusion": ["kalman_filter.py", "hybrid_fusion.py", "adaptive_weighting.py"],
    "training": ["train.py"],
    "training/trainers": ["base_trainer.py", "ensemble_trainer.py", "uncertainty_trainer.py"],
    "training/losses": ["nll_loss.py", "evidential_loss.py", "calibration_loss.py", "uncertainty_aware_loss.py"],
    "training/optimizers": ["lr_schedulers.py", "optimizer_factory.py"],
    "training/callbacks": ["uncertainty_metrics.py", "visualization.py", "early_stopping.py"],
    "evaluation": ["evaluate.py"],
    "evaluation/metrics": ["trajectory_metrics.py", "uncertainty_metrics.py", "calibration_metrics.py", "robustness_metrics.py"],
    "evaluation/visualization": ["trajectory_viz.py", "uncertainty_viz.py", "attention_viz.py"],
    "evaluation/benchmarks": ["baseline_comparison.py", "sota_comparison.py", "ablation_studies.py"],
    "inference": ["predict.py"],
    "inference/predictors": ["real_time_predictor.py", "ensemble_predictor.py", "rejection_predictor.py"],
    "inference/deployment": ["model_optimization.py", "onnx_export.py", "tensorrt_conversion.py"],
    "inference/integration": ["kalman_integration.py", "decision_making.py", "safety_module.py"],
    "configs": ["config_manager.py"],
    "configs/data_configs": ["nuscenes_config.yaml", "argoverse_config.yaml", "interaction_config.yaml"],
    "configs/model_configs": ["baseline_config.yaml", "ablation_config.yaml", "final_config.yaml"],
    "configs/training_configs": ["lr_configs.yaml", "loss_configs.yaml", "ensemble_configs.yaml"],
    "utils": ["helpers.py"],
    "utils/logging": ["experiment_logger.py", "tensorboard_logger.py", "wandb_logger.py"],
    "utils/analysis": ["uncertainty_analysis.py", "failure_analysis.py", "statistical_tests.py"],
    "utils/visualization": ["trajectory_plots.py", "uncertainty_plots.py", "interactive_viz.py"]
}

# Function to create directories and files
def create_project_structure(base_path, structure):
    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(folder_path, file)
            open(file_path, 'w').close()  # Create an empty file

# Define the base directory and create the project structure
base_directory = os.getcwd()  # Set to current working directory or change as needed
create_project_structure(base_directory, project_structure)

print(f"Project structure created successfully in {base_directory}")
