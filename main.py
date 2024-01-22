from trainer import TransformerTrainer
from dataset.translation_dataset import TranslationDataset

from torch.utils.data import Subset

if __name__ == "__main__":
    train_dataset = TranslationDataset('train')
    test_dataset = TranslationDataset('test')
    validation_dataset = TranslationDataset('validation')

    train_subset = Subset(train_dataset, range(0, 100000))

    trainer = TransformerTrainer(train_subset,
                                 test_dataset,
                                 validation_dataset,
                                 batch_size=1024,
                                 d_model=512,
                                 vocab_size=50000,
                                 max_seq_len=64,
                                 num_heads=2,
                                 num_decoder_layers=4,
                                 num_encoder_layers=4,
                                 dim_feedforward=64,
                                 dropout=0.0,
                                 src_pad_idx=0)
    
    model, dict_log = trainer.train(num_epochs=5)