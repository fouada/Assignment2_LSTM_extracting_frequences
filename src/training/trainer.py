"""
Professional Training Module
Handles the training loop with state management for stateful LSTM.

Author: Professional ML Engineering Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from ..models.lstm_extractor import StatefulLSTMExtractor
from ..data.dataset import StatefulDataLoader

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """
    Professional trainer for stateful LSTM with comprehensive features:
    - State management for L=1 training
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Tensorboard logging
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: StatefulLSTMExtractor,
        train_loader: StatefulDataLoader,
        val_loader: Optional[StatefulDataLoader],
        config: Dict,
        device: torch.device,
        experiment_dir: Path
    ):
        """
        Initialize the trainer.
        
        Args:
            model: LSTM model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Device for training
            experiment_dir: Directory for saving experiments
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(experiment_dir / 'tensorboard'))
        
        logger.info("LSTMTrainer initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Optimizer: {self.config['optimizer']}")
        logger.info(f"Initial LR: {self.config['learning_rate']}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config['optimizer'].lower()
        lr = float(self.config['learning_rate'])
        weight_decay = float(self.config.get('weight_decay', 0))
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', None)
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(self.config.get('scheduler_factor', 0.5)),
                patience=int(self.config.get('scheduler_patience', 5))
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(self.config['epochs']),
                eta_min=1e-6
            )
        elif scheduler_type is None:
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        return scheduler
    
    def train_epoch(self) -> float:
        """
        Train for one epoch with stateful processing.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.config['epochs']}",
            leave=False
        )
        
        for batch in pbar:
            # Extract batch data
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            is_first_batch = batch['is_first_batch']
            freq_idx = batch['freq_idx']
            
            # Reset state at the start of each frequency sequence
            if is_first_batch:
                self.model.reset_state()
                logger.debug(f"State reset for frequency {freq_idx}")
            
            # Forward pass
            outputs = self.model(inputs, reset_state=False)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip_value'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.config['gradient_clip_value'])
                )
            
            self.optimizer.step()
            
            # Detach state from computation graph (for TBPTT)
            self.model.detach_state()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
            # Log to tensorboard
            if self.global_step % self.config.get('log_frequency', 100) == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return float('nan')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            is_first_batch = batch['is_first_batch']
            
            # Reset state at the start of each frequency sequence
            if is_first_batch:
                self.model.reset_state()
            
            # Forward pass
            outputs = self.model(inputs, reset_state=False)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def should_early_stop(self, val_loss: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        patience = int(self.config.get('early_stopping_patience', 10))
        min_delta = float(self.config.get('min_delta', 1e-6))
        
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint(is_best=True)
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                return True
            return False
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config['epochs']}")
        logger.info(f"Batches per epoch: {len(self.train_loader)}")
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('val/epoch_loss', val_loss, epoch)
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # Print progress
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.2e}"
            )
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping check
            if self.val_loader is not None:
                if self.should_early_stop(val_loss):
                    break
        
        logger.info("Training completed!")
        self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.experiment_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.experiment_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved with val_loss={self.best_val_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

