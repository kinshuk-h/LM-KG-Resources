""" Pre-processing utilities. """

from . import basic, ner

import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_data(
    data,
    lowercase=True,
    expand_contractions=True,
    remove_special=True,
    remove_stopwords=True,
    stem=True,
    num_to_words=True,
    lemmatize=True,
    remove_abbreviations=True
):
    """
        Preprocess data
        return: preprocessed data
    """
    # convert to lowercase
    if lowercase:
        data = basic.convert_to_lowercase(data)
        # convert data to string
    #expand contradictions
    if expand_contractions:
        data = basic.expand_contradictions(str(data))
    # remove special characters
    if remove_special:
        data = basic.remove_special_characters(data)
    # remove single character words
    if remove_stopwords:
        data = basic.remove_stop_words(data)
    # stemming
    if stem:
        data = basic.stemming(data)
    # convert numbers to words
    if num_to_words:
        data = basic.convert_numbers_to_words(data)
    # lemmatization
    if lemmatize:
        data = basic.lemmatization(data)
    # remove abbriviation
    if remove_abbreviations:
        data = basic.remove_abbreviation(str(data))
    return str(data)

def preprocess_file(
    file_path,
    output_file_path=None,
    lowercase=True,
    expand_contractions=True,
    remove_special=True,
    remove_stopwords=True,
    stem=True,
    num_to_words=True,
    lemmatize=True,
    remove_abbreviations=True
):
    """
    Preprocess file
    return: preprocessed file

    """
    # read file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # preprocess data
    data = preprocess_data(
        data, lowercase, expand_contractions, remove_special, remove_stopwords,
        stem, num_to_words, lemmatize, remove_abbreviations
    )

    if output_file_path is not None:
        # write file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(data)

    return data

__all__ = [ "basic", "ner", "preprocess_data", "preprocess_file" ]
