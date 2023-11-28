from tokenizers.implementations import CharBPETokenizer
from transformers import GPT2Tokenizer

class Tokenizer():
    def __init__(self) -> None:
        self.gpt2_tokenizer = None
        
    def train(self, corpus_file):
        char_bpe_tokenizer = CharBPETokenizer(unk_token='[UNK]')
        char_bpe_tokenizer.train(corpus_file, 50000)

        vocab_dict = {token: idx for idx, token in enumerate(char_bpe_tokenizer.get_vocab())}
        merges_dict = {tuple(merge.split()): idx for idx, merge in enumerate(char_bpe_tokenizer.model)}

        self.gpt2_tokenizer = GPT2Tokenizer().from_pretrained(pretrained_model_name_or_path=None,
                                                            vocab_dict=dict(),
                                                            merges_dict=dict(),
                                                            unk_token='[UNK]',
                                                            bos_token='[BOS]',
                                                            eos_token='[EOS]',
                                                            pad_token='[PAD]')

        
    def encode(self, text):
        if self.gpt2_tokenizer is None:
            raise Exception('Tokenizer is not trained.')
        return self.gpt2_tokenizer.encode(text)