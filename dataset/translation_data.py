import torch
import json

from torch.utils.data import Dataset

from datasets import load_dataset

from dataset_utils import clean_pair


class TranslationDataset(Dataset):
    def __init__(self, tokenizer, mode) -> None:
        super().__init__()

        self.mode = mode
        # self.data = self.load_data()
        self.tokenizer = tokenizer
        self.tokenizer.train('resources/cleaned_train_pairs.json')

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
            cleaned_pair = clean_pair(pair['translation'])
            if cleaned_pair is not None:
                data.append(cleaned_pair)

        with open(f'resources/cleaned_{self.mode}_pairs.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False)

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
    

from tokenizer import Tokenizer


if __name__ == '__main__':
    tk = Tokenizer()
    tr_ds = TranslationDataset(tk, 'train')
    te_ds = TranslationDataset(tk, 'test')
    va_ds = TranslationDataset(tk, 'validation')

    torch.save(tr_ds, 'resources/train_dataset.pth')
    torch.save(te_ds, 'resources/test_dataset.pth')
    torch.save(va_ds, 'resources/validation_dataset.pth')
         