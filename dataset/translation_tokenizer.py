import json

from tokenizers import decoders, pre_tokenizers, processors, Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from transformers import GPT2Tokenizer

class TranslationTokenizer():
    def __init__(self, batch_size: int = 2048, vocab_size: int = 50000) -> None:
        self.gpt2_tokenizer = None

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        
    def train(self, corpus_file, generate_bpe_files: bool = False):

        if generate_bpe_files:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                corpus = f.read()

            corpus = corpus.split('\n')

            def batch_iterator():
                for i in range(0, len(corpus), self.batch_size):
                    yield corpus[i: i+self.batch_size]

            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

            special_tokens = ["[PAD]", "[EOS]", "[BOS]", "[UNK]"]
            trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=special_tokens)
            tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
            tokenizer.decoder = decoders.ByteLevel()

            model = json.loads(tokenizer.to_str())['model']
            vocab_dict = model['vocab']
            with open('resources/gpt2_from_pretrained/vocab.json', 'w', encoding='utf-8') as f:
                json.dump(vocab_dict, f, ensure_ascii=False)

            merges_list = model['merges']
            with open('resources/gpt2_from_pretrained/merges.txt', 'w',  encoding='utf-8') as f:
                for rule in merges_list:
                    f.write("%s\n" % rule)

        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./resources/gpt2_from_pretrained")
       
    def encode(self, text):
        if self.gpt2_tokenizer is None:
            raise Exception('Tokenizer is not trained.')
        return self.gpt2_tokenizer.encode(text)
    
    def encode_tokens(self, text):
        if self.tokenizer_gpt2 is None:
            raise Exception('Tokenizer is not trained.')
        return self.tokenizer_gpt2.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        if self.tokenizer_gpt2 is None:
            raise Exception('Tokenizer is not trained.')
        return self.tokenizer_gpt2.convert_tokens_to_ids(tokens)
