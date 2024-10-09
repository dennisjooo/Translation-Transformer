import os
import argparse
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from config import config
from src.dataset import DataModule
from src.lightning_module import TransformerLightning
from src.model import Transformer
from src.sampler import SamplerCallback

def main(args):
    # If a folder called tokenizer does not exist, create it
    if not os.path.exists("tokenizer"):
        os.makedirs("tokenizer")

    # Update config with command-line arguments
    config.update(vars(args))

    # Create the lightning model
    model = Transformer(config["n_embed"], config["n_head"], config["n_hidden"], config["n_layers"],
                        config["max_vocab"], config["max_len"], config["padding_value"], config["dropout_p"])

    # Use the xavier uniform initialization
    def init_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_normal_(m.weight)
            
    # Apply the initialization
    model.apply(init_weights)

    # Calculate the total_steps
    total_steps = config['max_epochs'] * (config['max_data'] // config['batch_size']) * config['split_size']

    warmup_steps = int(0.1 * total_steps)

    # Create the lightning model
    lightning_model = TransformerLightning(model, config["lr"], config["padding_value"], 
                                           total_steps, config['max_vocab'], config['lambda_val'], 
                                           warmup_steps, config['grad_accum_steps'])

    # Create the data module
    data_module = DataModule(config["data_path"], config["max_vocab"], config["batch_size"], 
                             config["max_len"], config["num_workers"], config["split_size"],
                             config["src_tokenizer_path"], config["tgt_tokenizer_path"])
    data_module.prepare_data(max_data=config['max_data'])

    # Setup model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='transformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Setup early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        verbose=True,
        mode='min'
    )

    # Setup some callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        SamplerCallback(model, data_module.src_tokenizer, data_module.tgt_tokenizer),
        checkpoint_callback,
        early_stopping_callback
    ]

    # Create the trainer
    trainer = L.Trainer(
        max_epochs=config["max_epochs"], 
        accelerator="cpu" if not torch.cuda.is_available() else "gpu",
        devices="auto",
        callbacks=callbacks,
        strategy='auto',
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config['grad_accum_steps'],
        check_val_every_n_epoch=1
    )

    # Fit the model
    trainer.fit(lightning_model, data_module)

    # Save the model
    torch.save(model.state_dict(), config["output_model_path"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for translation.")
    parser.add_argument("--data_path", type=str, default=config["data_path"], help="Path to the training data")
    parser.add_argument("--max_vocab", type=int, default=config["max_vocab"], help="Maximum vocabulary size")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="Batch size for training")
    parser.add_argument("--max_len", type=int, default=config["max_len"], help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=config["num_workers"], help="Number of workers for data loading")
    parser.add_argument("--split_size", type=float, default=config["split_size"], help="Train/val split ratio")
    parser.add_argument("--src_tokenizer_path", type=str, default=config["src_tokenizer_path"], help="Path to source tokenizer")
    parser.add_argument("--tgt_tokenizer_path", type=str, default=config["tgt_tokenizer_path"], help="Path to target tokenizer")
    parser.add_argument("--n_embed", type=int, default=config["n_embed"], help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=config["n_head"], help="Number of attention heads")
    parser.add_argument("--n_hidden", type=int, default=config["n_hidden"], help="Hidden layer dimension")
    parser.add_argument("--n_layers", type=int, default=config["n_layers"], help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=config["lr"], help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=config["max_epochs"], help="Maximum number of epochs")
    parser.add_argument("--max_data", type=int, default=config["max_data"], help="Maximum number of data points to use")
    parser.add_argument("--grad_accum_steps", type=int, default=config["grad_accum_steps"], help="Gradient accumulation steps")
    parser.add_argument("--lambda_val", type=float, default=config["lambda_val"], help="L2 regularization strength")
    parser.add_argument("--dropout_p", type=float, default=config["dropout_p"], help="Dropout probability")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Patience for early stopping")
    parser.add_argument("--output_model_path", type=str, default="transformer.pth", help="Path to save the trained model")

    args = parser.parse_args()
    main(args)