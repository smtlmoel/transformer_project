
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
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerTrainer(nn.Module):
    def __init__(self,
                 train_dataset: TranslationDataset,
                 test_dataset: TranslationDataset,
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
                 src_pad_idx: int,
                 ckpt_name: Union[str, os.PathLike] = "./resources/ckpts/checkpoint.pth") -> None:
        super().__init__()

        self.ckpt_name = Path(ckpt_name)

        self.epoch = 0

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
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

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = Transformer(vocab_size=self.vocab_size,
                                 d_model=self.d_model,
                                 num_heads=self.num_heads,
                                 num_encoder_layers=self.num_encoder_layers,
                                 num_decoder_layers=self.num_decoder_layers,
                                 dim_feedforward=self.dim_feedforward,
                                 dropout=self.dropout,
                                 max_len=self.max_seq_len)
        
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
        src_mask = (src_input != self.src_pad_idx).int()
        trg_mask = (trg_input != self.src_pad_idx).int()
        return src_mask, trg_mask

    def train_one_epoch(self):
        self.model.train()
        batch_loss = []
        for batch in self.train_dataloader:
            src_input, trg_input, trg_output = batch['source'].to(device), batch['target_input'].to(device), batch['target_output'].to(device)
            src_mask, trg_mask = self._make_masks(src_input, trg_input)

            output = self.model(src_input, src_mask, trg_input, trg_mask)

            self.optimizer.zero_grad()
            loss = self.ce_loss(output.view(-1, self.vocab_size), trg_output.view(-1))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_loss.append(loss.cpu().detach())

        return np.mean(batch_loss)
    
    def validation(self):
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.validation_dataloader)):
                src_input, trg_input, trg_output = batch['src'], batch['tgt_inp'], batch['tgt_out']
                src_mask, trg_mask = self._make_masks(src_input, trg_input, self.src_pad_idx)

                output = self.model(src_input, src_mask, trg_input, trg_mask)
                loss = self.ce_loss(output.view(-1, self.vocab_size), trg_output.view(-1))

                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save(self):
        torch.save(dict(
            model=self.model.state_dict(),
            optim=self.optimizer.state_dict(),
            epoch=self.epoch,
        ), self.ckpt_name)

    def load(self):
        print('Loading checkpoint', self.ckpt_name)
        ckpt = torch.load(self.ckpt_name)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        self.epoch = ckpt['epoch']

    def train(self, num_epochs: int, resume=False):
        self.epoch = 0

        if resume and self.ckpt_name.exists():
            self.load()

        dict_log = {"train_loss": [], "val_loss": []}
        best_loss = float('inf')

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            self.epoch = epoch
            train_loss = self.train_one_epoch()

            val_loss = self.validation()
            
            # Print epoch results to screen
            msg = (f'Ep {epoch}/{num_epochs}: || Loss: Train {train_loss:.3f} || Loss: Val {val_loss:.3f}')
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