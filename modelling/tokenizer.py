from collections import defaultdict

import torch
import torch.nn as nn

class BPETokenizer():
    def __init__(self, vocab_size=12) -> None:
        super().__init__()

        self.vocabulary = []
        self.merges = []

    def train(self, corpus):
        # Get all  unique words and count frequency
        unique_words_set = set()
        word_freqs = defaultdict(int)
        for sentence in corpus:
            words_in_sentence = sentence.split()
            for word in words_in_sentence:
                word_freqs[word] += 1
            unique_words_set = unique_words_set.union(set(words_in_sentence))
        unique_words_list = list(unique_words_set)
        
        # Get all unique base vocabulary
        base_vocabulary_set = set()
        for word in unique_words_list:
            base_vocabulary_set = base_vocabulary_set.union(word)
        base_vocabulary_list = list(base_vocabulary_set)




corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

tok = BPETokenizer()
tok.train(corpus)