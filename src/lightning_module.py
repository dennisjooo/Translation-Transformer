import lightning as L
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Tuple, Dict, Any, List


class TransformerLightning(L.LightningModule):
    """
    A PyTorch Lightning module for training and evaluating a Transformer model.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, lr: float = 1e-3, padding_value: int = 0, 
                 total_steps: int = 100, n_classes: int = 2, lambda_val: float = 0.01, 
                 warmup_steps: int = 1000, grad_accum_steps: int = 1):
        """
        Initialize the TransformerLightning module.

        Args:
            model (nn.Module): The Transformer model.
            tokenizer: The tokenizer for decoding predictions
            lr (float, optional): Learning rate. Defaults to 1e-3.
            padding_value (int, optional): Value used for padding. Defaults to 0.
            total_steps (int, optional): Total number of training steps. Defaults to 100.
            n_classes (int, optional): Number of classes for classification. Defaults to 2.
            lambda_val (float, optional): L2 regularization strength. Defaults to 0.01.
            warmup_steps (int, optional): Number of warmup steps for learning rate scheduler. 
            Defaults to 1000.
            grad_accum_steps (int, optional): Number of gradient accumulation steps. Defaults to 1.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.padding_value = padding_value
        self.total_steps = total_steps
        self.lambda_val = lambda_val
        self.warmup_steps = warmup_steps
        self.grad_accum_steps = grad_accum_steps

        # Initialize accuracy metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", 
                                               num_classes=n_classes,
                                               ignore_index=self.padding_value)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", 
                                             num_classes=n_classes,
                                             ignore_index=self.padding_value)
        
        # Initialize BLEU score parameters
        self.bleu_smoother = SmoothingFunction().method1
        self.bleu_weights = (0.5, 0.5, 0.0, 0.0)  # Use only unigrams and bigrams
        self.train_bleu_score = 0.0
        self.val_bleu_score = 0.0
        
        # Define loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_value, 
                                             label_smoothing=0.1)
        
    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                     batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]], List[List[str]]]:
        """
        Perform a shared step for both training and validation.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Input batch containing source,
            target, and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]], List[List[str]]]: Loss, predictions, and labels.
        """
        X_src, X_tgt, y = batch
        y_hat = self.model(X_src, X_tgt)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)
        
        # Calculate token-level loss
        loss = self.criterion(y_hat, y)
        
        # Reshape predictions and targets for BLEU calculation
        pred_tokens = y_hat.argmax(dim=-1).view(X_tgt.shape[0], -1)
        target_tokens = y.view(X_tgt.shape[0], -1)
        
        # Decode sequences and split into tokens
        pred_seqs = [
            self.tokenizer.decode(
                [t.item() for t in seq if t.item() != self.padding_value]
            ).split()  # Split into tokens after decoding
            for seq in pred_tokens
        ]
        
        target_seqs = [
            self.tokenizer.decode(
                [t.item() for t in seq if t.item() != self.padding_value]
            ).split()  # Split into tokens after decoding
            for seq in target_tokens
        ]
        
        return loss, y_hat, y, pred_seqs, target_seqs
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                      batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Input batch containing source, target, 
            and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.
        """
        loss, y_hat, y, pred_seqs, target_seqs = self._shared_step(batch, batch_idx)
        loss = loss / self.grad_accum_steps
        
        # Calculate metrics
        self.train_acc(y_hat, y)
        
        # Calculate average BLEU score - matching test implementation
        scores = [
            sentence_bleu([ref], hyp, weights=self.bleu_weights, smoothing_function=self.bleu_smoother)
            for ref, hyp in zip(target_seqs, pred_seqs)
        ]
        self.train_bleu_score = sum(scores) / len(scores) if scores else 0.0
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_bleu", self.train_bleu_score, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                        batch_idx: int) -> torch.Tensor:
        """
        Perform a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Input batch containing source, target, 
            and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The calculated loss.
        """
        loss, y_hat, y, pred_seqs, target_seqs = self._shared_step(batch, batch_idx)
        
        # Calculate metrics
        self.val_acc(y_hat, y)
        
        # Calculate average BLEU score - matching test implementation
        scores = [
            sentence_bleu([ref], hyp, weights=self.bleu_weights, smoothing_function=self.bleu_smoother)
            for ref, hyp in zip(target_seqs, pred_seqs)
        ]
        self.val_bleu_score = sum(scores) / len(scores) if scores else 0.0
        
        # Log metrics
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_bleu", self.val_bleu_score, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def get_lr_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler.LambdaLR:
        """
        Get the learning rate scheduler.

        Args:
            optimizer (optim.Optimizer): The optimizer to use with the scheduler.

        Returns:
            optim.lr_scheduler.LambdaLR The learning rate scheduler.
        """
        def lr_lambda(current_step):
            # Warmup phase: learning rate increases linearly to 1 (scaling factor)
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))

            # Cosine annealing phase: learning rate decreases from 1 to 0 (scaling factor)
            progress = ((current_step - self.warmup_steps) / 
                        float(max(1, self.total_steps - self.warmup_steps)))
            return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * progress)))

        # Return LambdaLR with the lambda function that scales the base lr
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Dict[str, Any]: A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), 
                                weight_decay=self.lambda_val)
        scheduler = self.get_lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}