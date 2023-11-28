import json

from tokenizers.implementations import CharBPETokenizer
from transformers import GPT2Tokenizer

class Tokenizer():
    def __init__(self) -> None:
        self.gpt2_tokenizer = None
        
    def train(self, corpus_file):
        char_bpe_tokenizer = CharBPETokenizer(unk_token='[UNK]')
        char_bpe_tokenizer.add_special_tokens(['[BOS]', '[EOS]', '[PAD]'])
        char_bpe_tokenizer.train(corpus_file, 50000)

        model = json.loads(char_bpe_tokenizer.to_str())['model']

        vocab_dict = model['vocab']
        with open(f'resources/vocab.json', 'w', encoding='utf-8') as file:
            json.dump(vocab_dict, file, ensure_ascii=False)

        merges_list = model['merges']
        with open('resources/merges.txt', 'w',  encoding='utf-8') as f:
            for rule in merges_list:
                f.write("%s\n" % rule)
        #merges_dict = {tuple(merge.split()): idx for idx, merge in enumerate(char_bpe_tokenizer.model)}

        self.gpt2_tokenizer = GPT2Tokenizer(vocab_file='resources/vocab.json',
                                            merges_file='resources/merges.txt').from_pretrained(pretrained_model_name_or_path='gpt2',
                                                            unk_token='[UNK]',
                                                            bos_token='[BOS]',
                                                            eos_token='[EOS]',
                                                            pad_token='[PAD]')

        
    def encode(self, text):
        if self.gpt2_tokenizer is None:
            raise Exception('Tokenizer is not trained.')
        return self.gpt2_tokenizer.encode(text)