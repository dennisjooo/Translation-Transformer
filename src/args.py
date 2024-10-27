import argparse
from config import config

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments for training a Transformer model.
    
    Sets up argument parsing for data, tokenizer, model architecture, training, and output parameters.
    Default values are pulled from a config dictionary.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with the following groups:
            Data arguments:
                data_path (str): Path to training data
                max_data (int): Maximum number of training examples to use
                split_size (float): Fraction of data to use for validation
                num_workers (int): Number of data loading workers
                
            Tokenizer arguments:
                max_vocab (int): Maximum vocabulary size
                src_tokenizer_path (str): Path to source language tokenizer
                tgt_tokenizer_path (str): Path to target language tokenizer
                
            Model architecture arguments:
                n_embed (int): Embedding dimension
                n_head (int): Number of attention heads
                n_hidden (int): Hidden layer dimension
                n_layers (int): Number of transformer layers
                max_len (int): Maximum sequence length
                dropout_p (float): Dropout probability
                
            Training arguments:
                batch_size (int): Training batch size
                lr (float): Learning rate
                max_epochs (int): Maximum training epochs
                grad_accum_steps (int): Gradient accumulation steps
                lambda_val (float): Loss function weighting parameter
                early_stopping_patience (int): Epochs before early stopping
                
            Output arguments:
                output_model_path (str): Path to save trained model
    """
    parser = argparse.ArgumentParser(description="Train a Transformer model for translation.")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default=config["data_path"])
    parser.add_argument("--max_data", type=int, default=config["max_data"])
    parser.add_argument("--split_size", type=float, default=config["split_size"])
    parser.add_argument("--num_workers", type=int, default=config["num_workers"])
    
    # Tokenizer arguments
    parser.add_argument("--max_vocab", type=int, default=config["max_vocab"])
    parser.add_argument("--src_tokenizer_path", type=str, default=config["src_tokenizer_path"])
    parser.add_argument("--tgt_tokenizer_path", type=str, default=config["tgt_tokenizer_path"])
    
    # Model architecture arguments
    parser.add_argument("--n_embed", type=int, default=config["n_embed"])
    parser.add_argument("--n_head", type=int, default=config["n_head"])
    parser.add_argument("--n_hidden", type=int, default=config["n_hidden"])
    parser.add_argument("--n_layers", type=int, default=config["n_layers"])
    parser.add_argument("--max_len", type=int, default=config["max_len"])
    parser.add_argument("--dropout_p", type=float, default=config["dropout_p"])
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--lr", type=float, default=config["lr"])
    parser.add_argument("--max_epochs", type=int, default=config["max_epochs"])
    parser.add_argument("--grad_accum_steps", type=int, default=config["grad_accum_steps"])
    parser.add_argument("--lambda_val", type=float, default=config["lambda_val"])
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    
    # Output arguments
    parser.add_argument("--output_model_path", type=str, default="transformer.pth")

    return parser.parse_args()
