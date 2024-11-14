import datetime
import logging
import os
import lightning as L
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import (
    EarlyStopping, 
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar
)
from lightning.pytorch.loggers import WandbLogger

from src.dataset import DataModule
from src.lightning_module import TransformerLightning
from src.model import Transformer
from src.sampler import SamplerCallback
from src.utils import init_weights

def setup_directories() -> None:
    """Create necessary directories for storing tokenizers and model checkpoints."""
    os.makedirs("tokenizer", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

def create_model(config: dict) -> nn.Module:
    """
    Create and initialize a transformer model with the given configuration.
    
    Args:
        config (dict): Model configuration parameters including architecture details
        
    Returns:
        nn.Module: Initialized transformer model
    """
    model = Transformer(
        config["n_embed"], config["n_head"], config["n_hidden"],
        config["n_layers"], config["max_vocab"], config["max_len"],
        config["padding_value"], config["dropout_p"]
    )
    model.apply(init_weights)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    return model

def setup_lightning_model(model: nn.Module, data_module: DataModule, config: dict) -> TransformerLightning:
    """
    Create a PyTorch Lightning wrapper for the transformer model.
    
    Args:
        model (nn.Module): The transformer model to wrap
        data_module (DataModule): The data module containing tokenizers
        config (dict): Training configuration parameters
        
    Returns:
        TransformerLightning: Lightning model wrapper
    """
    total_steps = config['max_epochs'] * (config['max_data'] // config['batch_size']) * config['split_size']
    return TransformerLightning(
        model=model,
        tokenizer=data_module.tgt_tokenizer,
        lr=config["lr"],
        padding_value=config["padding_value"],
        total_steps=total_steps,
        n_classes=config['max_vocab'],
        lambda_val=config['lambda_val'],
        warmup_steps=config['warmup_steps'],
        grad_accum_steps=config['grad_accum_steps']
    )

def setup_data_module(config: dict) -> DataModule:
    """
    Create and prepare the data module for training.
    
    Args:
        config (dict): Data configuration parameters
        
    Returns:
        DataModule: Prepared data module for training
    """
    data_module = DataModule(
        config["data_path"], config["max_vocab"], config["batch_size"],
        config["max_len"], config["num_workers"], config["split_size"],
        config["src_tokenizer_path"], config["tgt_tokenizer_path"]
    )
    data_module.prepare_data(max_data=config['max_data'])
    return data_module

def setup_wandb(model: nn.Module, config: dict) -> WandbLogger:
    """
    Initialize Weights & Biases logging.
    
    Args:
        model (nn.Module): Model to track
        config (dict): Configuration to log
        
    Returns:
        WandbLogger: Initialized W&B logger
        
    Raises:
        Exception: If W&B initialization fails
    """
    try:
        wandb.login(key=os.getenv("WANDB_KEY"))
        run = wandb.init(
            project="Translation Transformer",
            name=datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
            config=config
        )
        logger = WandbLogger(experiment=run)
        logger.watch(model, log="all")
        return logger
    except Exception as e:
        logging.error(f"Failed to initialize wandb: {e}")
        raise

def setup_callbacks(model: nn.Module, data_module: DataModule, config: dict) -> list:
    """
    Set up training callbacks for monitoring and controlling the training process.

    Args:
        model (nn.Module): The model being trained
        data_module (DataModule): Data module containing tokenizers
        config (dict): Configuration dictionary containing callback parameters

    Returns:
        list: List of callbacks including:
            - LearningRateMonitor: Logs learning rate at each step
            - SamplerCallback: Generates translation samples during training
            - ModelCheckpoint: Saves model checkpoints based on BLEU score
            - EarlyStopping: Stops training if BLEU score plateaus
            - TQDMProgressBar: Shows training progress
    """
    return [
        LearningRateMonitor(logging_interval='step'),
        SamplerCallback(
            model, 
            data_module.src_tokenizer, 
            data_module.tgt_tokenizer,
            every_n_steps=50  # Sample more frequently
        ),
        ModelCheckpoint(
            monitor='val_bleu',  # Monitor BLEU score instead of loss
            dirpath='checkpoints',
            filename='transformer-{epoch:02d}-{val_bleu:.2f}',
            save_top_k=3,
            mode='max',
            auto_insert_metric_name=True,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_bleu',  # Monitor BLEU score for early stopping
            patience=config['early_stopping_patience'],
            verbose=True,
            mode='max'
        ),
        TQDMProgressBar(refresh_rate=20)
    ]

def setup_trainer(config: dict, callbacks: list, wandb_logger: WandbLogger) -> L.Trainer:
    """
    Create a PyTorch Lightning trainer with the specified configuration.
    
    Args:
        config (dict): Training configuration parameters
        callbacks (list): List of callbacks to use during training
        wandb_logger (WandbLogger): W&B logger for experiment tracking
        
    Returns:
        L.Trainer: Configured Lightning trainer
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return L.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices="auto", 
        callbacks=callbacks,
        strategy='auto',
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config['grad_accum_steps'],
        val_check_interval=500,
        precision=config["precision"],
        max_time=datetime.timedelta(hours=18),
        logger=wandb_logger
    )
