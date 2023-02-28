""" Basic, elementary, composable operations """

import numpy as np
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from . import constants

def split_sentences(data):
    """
        Split sentences
        return: sentences
    """
    sentences = sent_tokenize(str(data))
    return sentences

def split_paragraphs(data):
    """
        Split paragraphs
        return: paragraphs
    """
    para_start = None
    for _match in constants.PARA_DEMARCATOR.finditer(data):
        if para_start is not None:
            yield constants.SPACE.sub(" ", data[para_start:_match.start()].strip())
        para_start = _match.end()
    yield constants.SPACE.sub(" ", data[para_start or 0:].strip())

def convert_to_lowercase(data):
    #convert string to lowercase
    data = data.lower()
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def remove_special_characters(data):
    """
    Remove all special characters
    return: data without special characters numpy array
    """
    #vineet irfan
    symbols = "!\"#$%&()*+-,/:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    data = remove_apostrophe(data)
    return data


def remove_stop_words(data):
    """
    Remove all stop words
    return: data without stop words strings
    """
    # remove all stop words
    stop_words = set(stopwords.words('english'))
    data = np.char.replace(data, '\s' + '|'.join(stop_words) + '\s', ' ')
    return data


def stemming(data):
    """
    Stemming
    return: words after stemming
    """
    stemmer= PorterStemmer()
  
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def convert_numbers_to_words(data):
    """
        Convert numbers to words
        return: data without numbers
    """
    # convert numbers to words
    # vineet-irfan
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            pass
        new_text += " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def lemmatization(data):
    """
        Lemmatization
        return: words after lemmatization
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w)
    return new_text

def expand_contractions(data):
    """
        Expand contractions
        return: data without contractions
    """
    # Checking for whether the given token matches with the Key & replacing word with key's value.
    # Check whether Word is in list_Of_tokens or not.
    # If Word is present in both dictionary & list_Of_tokens, replace that word with the key value.
    for contraction, expansion in constants.CONTRACTION_DICT.items():
        data = contraction.sub(expansion, data)
    return data.strip()

def remove_abbreviation(data):
    """ Removes abbreviations present in source text.

    Args:
        data (str): Source text data

    Returns:
        str: Text with expanded abbreviations.
    """
    # Riya Payal
    for abbr, expansion in constants.ABBREVIATION_DICT.items():
        data = abbr.sub(expansion, data)
    return data.strip()
