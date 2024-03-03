from trainer import TransformerTrainer
from tester import TransformerTester
from plotting import plot_losses
from dataset.translation_dataset import TranslationDataset
from torch.utils.data import Subset
import torch

def train(train_dataset,
        validation_dataset,
        epochs,
        batch_size,
        d_model,
        vocab_size,
        max_seq_len,
        num_heads,
        num_decoder_layers,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        warmup_steps,
        src_pad_idx):

    trainer = TransformerTrainer(train_dataset,
                                 validation_dataset,
                                 batch_size=batch_size,
                                 d_model=d_model,
                                 vocab_size=vocab_size,
                                 max_seq_len=max_seq_len,
                                 num_heads=num_heads,
                                 num_decoder_layers=num_decoder_layers,
                                 num_encoder_layers=num_encoder_layers,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout,
                                 warmup_steps=warmup_steps,
                                 src_pad_idx=src_pad_idx)
    
    return trainer.train(num_epochs=epochs)

def test(test_dataset,
        d_model,
        vocab_size,
        max_seq_len,
        num_heads,
        num_decoder_layers,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        src_pad_idx,
        trg_eos_idx,
        trg_sos_idx,
        ckpt_name):

    tester = TransformerTester(test_dataset=test_dataset,
                               d_model=d_model,
                               vocab_size=vocab_size,
                               max_seq_len=max_seq_len,
                               num_heads=num_heads,
                               num_decoder_layers=num_decoder_layers,
                               num_encoder_layers=num_encoder_layers,
                               dim_feedforward=dim_feedforward,
                               dropout=dropout,
                               src_pad_idx=src_pad_idx,
                               trg_eos_idx=trg_eos_idx,
                               trg_sos_idx=trg_sos_idx,
                               ckpt_name=ckpt_name)
    
    tester.test()

if __name__ == "__main__":
    train_dataset = TranslationDataset('train')
    test_dataset = TranslationDataset('test')
    validation_dataset = TranslationDataset('validation')

    train_subset = Subset(train_dataset, range(0, 2000000))

    epochs=5
    batch_size=256
    d_model=128
    vocab_size=50000
    max_seq_len=64
    num_heads=4
    num_decoder_layers=4
    num_encoder_layers=4
    dim_feedforward=64
    dropout=0.1
    warmup_steps=3900 # Ratio (0.1) * ((Samples (2000000) / Batch-Size (256)) * epochs (5))
    src_pad_idx=0
    trg_eos_idx=1
    trg_sos_idx=2

    _, dict_log = train(train_dataset=train_subset,
                        validation_dataset=validation_dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        d_model=d_model,
                        vocab_size=vocab_size,
                        max_seq_len=max_seq_len,
                        num_heads=num_heads,
                        num_decoder_layers=num_decoder_layers,
                        num_encoder_layers=num_encoder_layers,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        warmup_steps=warmup_steps,
                        src_pad_idx=src_pad_idx)
    
    torch.save(dict_log, './resources/results/dict_log.pth')
    
    plot_losses(dict_log['train_loss'], dict_log['val_loss'], 'Loss_per_epoch.png')

    ckpt_name = f'./resources/ckpts/checkpoint_epoch={epochs}.pth'

    test(test_dataset=test_dataset,
         d_model=d_model,
         vocab_size=vocab_size,
         max_seq_len=max_seq_len,
         num_heads=num_heads,
         num_decoder_layers=num_decoder_layers,
         num_encoder_layers=num_encoder_layers,
         dim_feedforward=dim_feedforward,
         dropout=dropout,
         src_pad_idx=src_pad_idx,
         trg_eos_idx=trg_eos_idx,
         trg_sos_idx=trg_sos_idx,
         ckpt_name=ckpt_name)