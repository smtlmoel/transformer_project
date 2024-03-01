from modelling.transformer import Transformer
from dataset.translation_dataset import TranslationDataset
from scheduler import TransformerLRScheduler

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Union
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerTrainer(nn.Module):
    """
    Transformer trainer module for training and validating Transformer models.

    Args:
        train_dataset (TranslationDataset): Training dataset.
        test_dataset (TranslationDataset): Testing dataset.
        validation_dataset (TranslationDataset): Validation dataset.
        batch_size (int): Batch size for training.
        d_model (int): Dimensionality of model.
        vocab_size (int): Size of vocabulary.
        max_seq_len (int): Maximum sequence length.
        num_heads (int): Number of attention heads.
        num_decoder_layers (int): Number of decoder layers.
        num_encoder_layers (int): Number of encoder layers.
        dim_feedforward (int): Dimensionality of feedforward layer.
        dropout (float): Dropout probability.
        src_pad_idx (int): Padding index for source sequence.
        ckpt_name (Union[str, os.PathLike]): Path to save checkpoints.
    """
    def __init__(self,
                 train_dataset: TranslationDataset,
                 validation_dataset: TranslationDataset,
                 batch_size: int,
                 d_model: int,
                 vocab_size: int,
                 max_seq_len: int,
                 num_heads: int,
                 num_decoder_layers: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 src_pad_idx: int) -> None:
        super().__init__()

        self.ckpt_path = "./resources/ckpts/"

        self.epoch = 0

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.src_pad_idx = src_pad_idx

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.scaler = GradScaler()

        self.model = Transformer(vocab_size=self.vocab_size,
                                 d_model=self.d_model,
                                 num_heads=self.num_heads,
                                 num_encoder_layers=self.num_encoder_layers,
                                 num_decoder_layers=self.num_decoder_layers,
                                 dim_feedforward=self.dim_feedforward,
                                 dropout=self.dropout,
                                 max_len=self.max_seq_len).to(device)
        
        optimizer_parameters = [
            {'params': [param for name, param in self.model.named_parameters()
                        if 'bias' in name or 'layer_norm' in name], 'weight_decay':0.0},
            {'params': [param for name, param in self.model.named_parameters()
                        if 'bias' not in name and 'layer_norm' not in name], 'weight_decay': 1e-2}
        ]
   
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)
        self.optimizer = AdamW(optimizer_parameters)
        self.scheduler = TransformerLRScheduler(optimizer=self.optimizer, d_model=self.d_model, warmup_steps=4000)

    def _make_masks(self, src_input, trg_input):
        """
        Create masks for source and target sequences.

        Args:
            src_input (torch.Tensor): Source input tensor.
            trg_input (torch.Tensor): Target input tensor.

        Returns:
            torch.Tensor: Source mask tensor.
            torch.Tensor: Target mask tensor.
        """
        src_mask = (src_input != self.src_pad_idx).int()
        trg_mask = (trg_input != self.src_pad_idx).int()
        return src_mask, trg_mask

    def train_one_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            float: Average training loss.
        """
        self.model.train()
        batch_loss = []
        pbar = tqdm(enumerate(self.train_dataloader), leave=False)
        for i, batch in pbar:

            msg = (f'Training: {i+1}/{len(self.train_dataloader)}')
            pbar.set_description(msg)

            src_input, trg_input, trg_output = batch['source'].to(device), batch['target_input'].to(device), batch['target_output'].to(device)
            src_mask, trg_mask = self._make_masks(src_input, trg_input)
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = self.model(src_input, src_mask, trg_input, trg_mask)
                loss = self.ce_loss(output.view(-1, self.vocab_size), trg_output.view(-1))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            batch_loss.append(loss.cpu().detach())

        return np.mean(batch_loss)
    
    def validation(self):
        """
        Perform validation on the model.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            pbar = tqdm(enumerate(self.validation_dataloader), leave=False)
            for i, batch in pbar:
                msg = (f'Validation: {i+1}/{len(self.validation_dataloader)}')
                pbar.set_description(msg)

                src_input, trg_input, trg_output = batch['source'].to(device), batch['target_input'].to(device), batch['target_output'].to(device)
                src_mask, trg_mask = self._make_masks(src_input, trg_input)
                src_mask = src_mask.to(device)
                trg_mask = trg_mask.to(device)

                output = self.model(src_input, src_mask, trg_input, trg_mask)
                loss = self.ce_loss(output.view(-1, self.vocab_size), trg_output.view(-1))

                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save(self):
        """Save the model."""
        torch.save(dict(
            model=self.model.state_dict(),
            optim=self.optimizer.state_dict(),
            epoch=self.epoch,
        ), Path(f'{self.ckpt_path}checkpoint_epoch={self.epoch}.pth'))

    def load(self, checkpoint_name):
        """Load the model."""
        path = Path(f'{self.ckpt_path}{checkpoint_name}')
        print('Loading checkpoint', path)
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        self.epoch = ckpt['epoch']

    def train(self, num_epochs: int, resume=False, checkpoint_name=None):
        """
        Train the model for a given number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
            resume (bool): Whether to resume training from a saved checkpoint.

        Returns:
            Tuple[nn.Module, dict]: Trained model and training logs.
        """
        self.epoch = 0

        if resume and checkpoint_name is not None:
            self.load()

        dict_log = {"train_loss": [], "val_loss": []}
        best_loss = float('inf')

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            self.epoch = epoch
            train_loss = self.train_one_epoch()

            val_loss = self.validation()
            
            # Print epoch results to screen
            msg = (f'Ep {epoch+1}/{num_epochs}: || Loss: Train {train_loss:.3f} || Loss: Val {val_loss:.3f}')
            pbar.set_description(msg)

            dict_log["train_loss"].append(train_loss)
            dict_log["val_loss"].append(val_loss)

            if val_loss < best_loss:
                self.save()
                best_loss = val_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss
                }

        return self.model, dict_log