from collections import defaultdict

class BPETokenizer():
    def __init__(self, vocab_size=12) -> None:
        super().__init__()
        self.vocab_size=vocab_size

        self.word_freqs = defaultdict(int)
        self.vocabulary = []
        self.merges = {}

    def _compute_pair_freqs(self, splits):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def _merge_pair(self, a, b, splits):
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def train(self, corpus):
        # Get all  unique words and count frequency
        unique_words_set = set()
        for sentence in corpus:
            words_in_sentence = sentence.split()
            for word in words_in_sentence:
                self.word_freqs[word] += 1
            unique_words_set = unique_words_set.union(set(words_in_sentence))
        unique_words_list = list(unique_words_set)
        
        # Get all unique base vocabulary
        base_vocabulary_set = set()
        for word in unique_words_list:
            base_vocabulary_set = base_vocabulary_set.union(word)
        self.vocabulary = list(base_vocabulary_set)

        splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        while len(self.vocabulary) < self.vocab_size:
            pair_freqs = self._compute_pair_freqs(splits)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            splits = self._merge_pair(*best_pair, splits)
            merge = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merge
            self.vocabulary.append(merge)
        
    def tokenize(self, text: str):
        splits = [[l for l in word] for word in text.split()]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split

        for split in splits:
            for i, tkn in enumerate(split):
                if tkn not in self.vocabulary:
                    split[i] = '[UNK]'

        return sum(splits, [])
