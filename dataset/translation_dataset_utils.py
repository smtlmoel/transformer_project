import re

def remove_html_tags(sentence: str):
    """
    Remove HTML tags from the input sentence.

    Args:
        sentence (str): Input sentence containing HTML tags.

    Returns:
        str: Sentence with HTML tags removed.
    """
    regex = re.compile(r'<.*?>')
    return re.sub(regex, '', sentence)

def remove_urls(sentence: str):
    """
    Remove URLs from the input sentence.

    Args:
        sentence (str): Input sentence containing URLs.

    Returns:
        str: Sentence with URLs removed.
    """
    regex = re.compile(r'https?:\/\/.*[\r\n]*')
    return re.sub(regex, '', sentence)

def remove_non_utf8(sentence: str):
    """
    Remove non-UTF-8 characters from the input sentence.

    Args:
        sentence (str): Input sentence containing non-UTF-8 characters.

    Returns:
        str: Sentence with non-UTF-8 characters removed.
    """
    return sentence.encode('utf-8', 'ignore').decode('utf-8')

def remove_not_whitelisted_chars(sentence: str):
    """
    Remove characters not whitelisted in the provided whitelist from the input sentence.

    Args:
        sentence (str): Input sentence.

    Returns:
        str: Sentence with non-whitelisted characters removed.
    """
    whitelist = r"abcdefghijklmnopqrstuvwxyz ÄÖÜäöüßABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    regex = re.compile(f'[^{re.escape(whitelist)}]')
    return re.sub(regex, '', sentence)

def clean_sentence(sentence: str):
    """
    Clean the input sentence by removing HTML tags, URLs, non-UTF-8 characters, and non-whitelisted characters.
    Also return the sentence in lower cases.

    Args:
        sentence (str): Input sentence.

    Returns:
        str: Cleaned sentence.
    """
    cleaned_utf8 = remove_non_utf8(sentence)
    cleaned_url = remove_urls(cleaned_utf8)
    cleaned_html = remove_html_tags(cleaned_url)
    cleaned_text = remove_not_whitelisted_chars(cleaned_html)

    return cleaned_text.lower()

def clean_pair(pair: dict, max_ratio: float = 5.0, min_len: int = 5, max_len: int = 64):
    """
    Clean a pair of sentences by removing HTML tags, URLs, non-UTF-8 characters, and non-whitelisted characters, 
    and then filter based on maximum ratio and minimum/maximum length constraints.

    Args:
        pair (dict): Dictionary containing 'de' (German) and 'en' (English) sentences.
        max_ratio (float): Maximum ratio between source and target sentence.
        min_len (int): Minimum length of sequences to consider.
        max_len (int): Maximum length of sequences to consider.

    Returns:
        dict: Dictionary containing cleaned 'de' and 'en' sentences, or None if the pair does not meet the constraints.
    """
    de = pair['de']
    en = pair['en']

    cleaned_de = clean_sentence(de)
    cleaned_en = clean_sentence(en)

    if (len(cleaned_de) < min_len) or (len(cleaned_de) > max_len) or (len(cleaned_en) < min_len) or (len(cleaned_en) > max_len):
        return None
    
    if max((len(cleaned_de)/len(cleaned_en)), (len(cleaned_en)/len(cleaned_de))) > max_ratio:
        return None
    
    return {'de': cleaned_de, 'en': cleaned_en}

def clean_dataset(dataset, max_ratio: float = 5.0, min_len: int = 5, max_len: int = 64):
    """
    Clean the entire dataset by applying clean_pair function to each pair.

    Args:
        dataset (list): List of dictionaries containing 'de' (German) and 'en' (English) sentences.
        max_ratio (float): Maximum ratio between source and target sentence.
        min_len (int): Minimum length of sequences to consider.
        max_len (int): Maximum length of sequences to consider.

    Returns:
        list: List of cleaned dictionaries containing 'de' and 'en' sentences.
    """
    cleaned_dataset = []

    for pair in dataset:
        cleaned_pair = clean_pair(pair, max_ratio=max_ratio, min_len=min_len, max_len=max_len)
        if cleaned_pair is not None:
            cleaned_dataset.append(cleaned_pair)

    return cleaned_dataset
