import re
import torch

from torch.utils.data import Dataset

from datasets import load_dataset
from tokenizers.implementations import CharBPETokenizer
from transformers import GPT2Tokenizer


class Tokenizer():
    def __init__(self, corpus) -> None:
        bpe_tokenizer = BPETokenizer(50000)
        bpe_tokenizer.train(corpus)

        vocab_dict = {token: idx for idx, token in enumerate(bpe_tokenizer.vocabulary)}
        merges_dict = {tuple(merge.split()): idx for idx, merge in enumerate(bpe_tokenizer.merges)}

        self.gpt2_tokenizer = GPT2Tokenizer().from_pretrained(pretrained_model_name_or_path='gpt2', vocab_dict=vocab_dict, merges_dict=merges_dict)

    def tokenize(self, text):
        return self.gpt2_tokenizer.tokenize(text)

    def encode(self, text):
        return self.gpt2_tokenizer.encode(text)
    

class TranslationDataset(Dataset):
    def __init__(self, mode, tokenizer):
        self.tokenizer = tokenizer
        self.mode = mode
        self.data = self.load_data()

    def load_data(self): 
        dataset = load_dataset("wmt17", "de-en")
        loaded_data = None
        if self.mode == 'train':
            loaded_data = dataset['train']
        elif self.mode == 'validation':
            loaded_data = dataset['validation']
        elif self.mode == 'test':
            loaded_data = dataset['test']
        else:
            loaded_data = None

        data = []
        for pair in loaded_data:
            cleaned_pair = clean_pair(pair)
            if cleaned_pair is not None:
                data.append(cleaned_pair)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        encoded_source = self.tokenizer.encode(sample['de'])
        encoded_target = self.tokenizer.encode(sample['en'])

        source_tensor = torch.tensor(encoded_source, dtype=torch.long)
        target_tensor = torch.tensor(encoded_target, dtype=torch.long)

        return {
            'source': source_tensor,
            'target': target_tensor
        }
    

if __name__ == '__main__':
    print()
