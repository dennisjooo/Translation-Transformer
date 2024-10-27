import lightning as L
import logging
import torch
import torch.nn as nn
import wandb
from typing import Any, Dict

def init_weights(m: nn.Module) -> None:
    """Initialize network weights using Xavier normal initialization.
    
    Applies Xavier normal initialization to any module that has a weight attribute
    with more than 1 dimension.
    
    Args:
        m: PyTorch module whose weights will be initialized
    """
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_normal_(m.weight)

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters for model training.
    
    Checks that all configuration parameters meet required constraints:
    - Embedding dimension must be positive
    - Number of attention heads must be positive 
    - Embedding dimension must be divisible by number of heads
    - Dropout probability must be between 0 and 1
    - Batch size must be positive
    - Maximum sequence length must be positive
    - Number of layers must be positive
    
    Args:
        config: Dictionary containing model configuration parameters
        
    Raises:
        AssertionError: If any configuration parameter is invalid
    """
    assert config["n_embed"] > 0, "Embedding dimension must be positive"
    assert config["n_head"] > 0, "Number of heads must be positive"
    assert config["n_embed"] % config["n_head"] == 0, "Embedding dimension must be divisible by number of heads"
    assert 0 <= config["dropout_p"] <= 1, "Dropout probability must be between 0 and 1"
    assert config["batch_size"] > 0, "Batch size must be positive"
    assert config["max_len"] > 0, "Maximum sequence length must be positive"
    assert config["n_layers"] > 0, "Number of layers must be positive"

def save_model(model: nn.Module, trainer: 'L.Trainer', config: dict) -> None:
    """Save the trained model's state dict and configuration.
    
    Saves a checkpoint containing:
    - Model state dictionary
    - Full configuration parameters
    - Current training epoch
    
    Args:
        model: Trained PyTorch model to save
        trainer: Lightning trainer instance containing training state
        config: Model configuration parameters
    """
    save_path = config["output_model_path"]
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': trainer.current_epoch,
    }, save_path)
    logging.info(f"Model saved to {save_path}")

def handle_training_error(error: Exception) -> None:
    """Handle training errors by logging and cleaning up wandb.
    
    Logs the error message and ensures the Weights & Biases run is properly
    terminated before re-raising the error.
    
    Args:
        error: Exception that occurred during training
        
    Raises:
        Exception: Re-raises the input error after handling
    """
    logging.error(f"Training failed: {error}")
    if wandb.run is not None:
        wandb.finish(exit_code=1)
    raise error
