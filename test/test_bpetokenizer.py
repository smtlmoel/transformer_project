from tokenizers.implementations import CharBPETokenizer

from modelling.BPETokenizer import BPETokenizer

def test_tokenizer():
    corpus = [
        "Machine learning helps in understanding complex patterns.",
        "Learning machine languages can be complex yet rewarding.",
        "Natural language processing unlocks valuable insights from data.",
        "Processing language naturally is a valuable skill in machine learning.",
        "Understanding natural language is crucial in machine learning."
    ]

    test_text = "Machine learning is a subset of artificial intelligence."

    tokenizer_one = CharBPETokenizer()
    tokenizer_one.train('test/text.txt', vocab_size=64)
    encoded_one = tokenizer_one.encode(test_text)

    tokenizer_two = BPETokenizer(vocab_size=64)
    tokenizer_two.train(corpus)
    encoded_two = tokenizer_two.tokenize(test_text)

    print(encoded_one.tokens)
    print(encoded_two)
    print()