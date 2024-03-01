import torch
import json
import os

from torch.utils.data import Dataset

from datasets import load_dataset

from dataset.translation_dataset_utils import clean_dataset
from dataset.translation_tokenizer import TranslationTokenizer


class TranslationDataset(Dataset):
    """
    Dataset class for translation tasks.

    Args:
        mode (str): Mode of the dataset, one of 'train', 'validation', 'test', or 'generate'.
        max_ratio (float): Maximum ratio between source and target sentences.
        min_len (int): Minimum length of sequences to consider.
        max_len (int): Maximum length of sequences to consider.
        tokenizer_batch_size (int): Batch size for training the tokenizer.
        tokenizer_vocab_size (int): Vocabulary size for the tokenizer.
        generate_bpe_files (bool): Whether to generate Byte Pair Encoding (BPE) files.
    """
    def __init__(self, mode, max_ratio: float = 5, min_len: int = 5, max_len: int = 64, tokenizer_batch_size: int = 2048, tokenizer_vocab_size: int = 50000, generate_bpe_files: bool = False) -> None:
        super().__init__()

        self.mode = mode

        self.max_ratio = max_ratio
        self.min_len = min_len
        self.max_len = max_len

        self.data = self.load_data()

        self.tokenizer_batch_size = tokenizer_batch_size
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.generate_bpe_files = generate_bpe_files

        self.tokenizer = TranslationTokenizer(self.tokenizer_batch_size, self.tokenizer_vocab_size)
        self.tokenizer.train(f'./resources/dataset/wmt17_cleaned_{self.mode}_pairs.txt', self.generate_bpe_files)

    def load_data(self):
        """
        Loads and cleans the dataset (only on the first run) based on the mode.

        Returns:
            list: List of dictionaries containing source and target sentences.
        """
        dataset = load_dataset("wmt17", "de-en")
        loaded_data = None
        if self.mode == 'train':
            loaded_data = dataset['train']['translation']
        elif self.mode == 'validation':
            loaded_data = dataset['validation']['translation']
        elif self.mode == 'test':
            loaded_data = dataset['test']['translation']
        elif self.mode == 'generate':
            loaded_data = dataset['train']['translation']
            val_translations = dataset['validation']['translation']
            # Extend train_translations with val_translations
            loaded_data.extend(val_translations)
        else:
            loaded_data = None

        output_path_json = f'./resources/dataset/wmt17_cleaned_{self.mode}_pairs.json'
        output_path_txt = f'./resources/dataset/wmt17_cleaned_{self.mode}_pairs.txt'

        # Check if the file already exists
        if not os.path.exists(output_path_json) and not os.path.exists(output_path_txt):
            cleaned_dataset = clean_dataset(loaded_data, self.max_ratio, self.min_len, self.max_len)

            values_list = [list(d.values()) for d in cleaned_dataset]
            sentences_list = [item for sublist in values_list for item in sublist]

            with open(output_path_txt, 'wb') as file:
                for sentence in sentences_list:
                    file.write(sentence.encode('utf-8') + '\n'.encode('utf-8'))

            with open(output_path_json, 'w', encoding='utf-8') as file:
                json.dump(cleaned_dataset, file, ensure_ascii=False)
        else:
            with open(output_path_json, 'r', encoding='utf-8') as file:
                cleaned_dataset = json.load(file)

        return cleaned_dataset
    
    def _add_padding_or_truncate(self, tokenized_sentence):
        """
        Adds padding or truncates the tokenized sentence based on the maximum length.

        Args:
            tokenized_sentence (list): Tokenized sentence as list.

        Returns:
            list: Tokenized sentence with padding or truncated to maximum length.
        """
        if len(tokenized_sentence) < self.max_len:
            left = self.max_len - len(tokenized_sentence)
            padding = [self.tokenizer.convert_tokens_to_ids("[PAD]")] * left
            tokenized_sentence += padding
        else:
            tokenized_sentence = tokenized_sentence[:self.seq_len]

        return tokenized_sentence

    
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing source, target input, and target output tensors.
        """
        sample = self.data[idx]

        encoded_source_without_padding = self.tokenizer.encode(sample['de'])
        encoded_target_without_padding = self.tokenizer.encode(sample['en'])

        encoded_source = self._add_padding_or_truncate(encoded_source_without_padding)

        encoded_target_input_without_padding = [self.tokenizer.convert_tokens_to_ids("[BOS]")] + encoded_target_without_padding
        encoded_target_output_without_padding = encoded_target_without_padding + [self.tokenizer.convert_tokens_to_ids("[EOS]")]

        encoded_target_input = self._add_padding_or_truncate(encoded_target_input_without_padding)
        encoded_target_output = self._add_padding_or_truncate(encoded_target_output_without_padding)

        source_tensor = torch.tensor(encoded_source, dtype=torch.long)
        target_input_tensor = torch.tensor(encoded_target_input, dtype=torch.long)
        target_output_tensor = torch.tensor(encoded_target_output, dtype=torch.long)

        return {
            'source': source_tensor,
            'target_input': target_input_tensor,
            'target_output': target_output_tensor
        }
    

###########################
#     Create datasets     #
###########################

from transformers import GPT2Tokenizer

if __name__ == '__main__':
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.save_pretrained("./resources/gpt2_from_pretrained")

    special_tokens_map_path = './resources/gpt2_from_pretrained/special_tokens_map.json'
    with open(special_tokens_map_path, 'r') as file:
        special_tokens_map = json.load(file)

    special_tokens_map['bos_token']['content'] = '[BOS]'
    special_tokens_map['eos_token']['content'] = '[EOS]'
    special_tokens_map['unk_token']['content'] = '[UNK]'

    with open(special_tokens_map_path, 'w', encoding='utf-8') as file:
        json.dump(special_tokens_map, file, ensure_ascii=False)

    TranslationDataset('generate', generate_bpe_files=True)

    train_dataset = TranslationDataset('train')
    test_dataset = TranslationDataset('test')
    validation_dataset = TranslationDataset('validation')

    torch.save(train_dataset, './resources/dataset/train_dataset.pth')
    torch.save(test_dataset, './resources/dataset/test_dataset.pth')
    torch.save(validation_dataset, './resources/dataset/validation_dataset.pth')