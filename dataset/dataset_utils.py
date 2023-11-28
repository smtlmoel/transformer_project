import re

def remove_html_tags(sentence: str):
    regex = re.compile(r'<.*?>')
    return re.sub(regex, '', sentence)

def remove_urls(sentence: str):
    regex = re.compile(r'https?:\/\/.*[\r\n]*')
    return re.sub(regex, '', sentence)

def remove_non_utf8(sentence: str):
    return sentence.encode('utf-8', 'ignore').decode('utf-8')

def remove_not_whitelisted_chars(sentence: str):
    whitelist = r"abcdefghijklmnopqrstuvwxyz ÄÖÜäöüßABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    regex = re.compile(f'[^{re.escape(whitelist)}]')
    return re.sub(regex, '', sentence)

def clean_sentence(sentence: str):
    cleaned_utf8 = remove_non_utf8(sentence)
    cleaned_url = remove_urls(cleaned_utf8)
    cleaned_html = remove_html_tags(cleaned_url)
    cleaned_text = remove_not_whitelisted_chars(cleaned_html)

    return cleaned_text.lower()

def clean_pair(pair: dict, max_ratio: float = 5.0, min_len: int = 5, max_len: int = 64):
    de = pair['de']
    en = pair['en']

    cleaned_de = clean_sentence(de)
    cleaned_en = clean_sentence(en)

    if (len(cleaned_de) < min_len) or (len(cleaned_de) > max_len) or (len(cleaned_en) < min_len) or (len(cleaned_en) > max_len):
        return None
    
    if max((len(cleaned_de)/len(cleaned_en)), (len(cleaned_en)/len(cleaned_de))) > max_ratio:
        return None
    
    return {'de': cleaned_de, 'en': cleaned_en}


#if __name__ == '__main__':
#    s1 = 'The latest news about artificial intelligence can be found at https://www.example.com/ai-news.'
#    s2 = 'The main heading of the webpage is enclosed in <h1> tags, and the paragraphs are wrapped in <p> tags for better structure.'
#    s3 = 'The café façade in São Paulo features décor with résumé flair.'
#    s4 = 'Jørn\'s résumé, filled with façade designs, is truly a pièce de résistance!'
#    c1 = clean_sentence(s1)
#    c2 = clean_sentence(s2)
#    c3 = clean_sentence(s3)
#    c4 = clean_sentence(s4)