import logging

from gensim import corpora
from gensim.matutils import corpus2csc
from gensim.models import TfidfModel
from gensim.parsing.porter import PorterStemmer
from gensim.utils import simple_preprocess
import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD

MIN_BOW_VALUE = -10
MAX_BOW_VALUE = 10


def preprocess_text(text: str) -> str:
    """Preprocess raw text by tokenizing, lowercasing, and removing punctuation"""
    tokens = simple_preprocess(text, deacc=True, min_len=2)
    return " ".join(tokens)


def stem_text(text: str) -> str:
    """Apply stemming to the input text"""
    stemmer = PorterStemmer()
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


def create_dictionary(texts: list[str]) -> corpora.Dictionary:
    """Create a Gensim dictionary from a list of texts"""
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)

    top_n = 10
    top_words = sorted(dictionary.cfs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    # Convert ids to words
    top_words_readable = [(dictionary[id_], freq) for id_, freq in top_words]

    logging.info(f"Created dictionary with {len(dictionary)} unique tokens")
    logging.info(f"Top tokens: {top_words_readable}")
    return dictionary


def text_to_bow(text: str, dictionary: corpora.Dictionary) -> list[tuple[int, int]]:
    """Convert preprocessed text to Bag-of-Words representation using the provided dictionary"""
    tokens = text.split()
    bow_vector = dictionary.doc2bow(tokens)
    return bow_vector


def texts_to_bow(texts: list[str], dictionary: corpora.Dictionary) -> list[list[tuple[int, int]]]:
    """Convert a list of preprocessed texts to their Bag-of-Words representations"""
    bow_corpus = [text_to_bow(text, dictionary) for text in texts]
    return bow_corpus


def texts_to_sparse_matrix(
    bow_corpus: list[list[tuple[int, int]]], dictionary: corpora.Dictionary, clip: bool = False
) -> np.ndarray:
    """Convert a list of preprocessed texts to their sparse matrix representations
    with TF-IDF weighting
    """
    tfidf_model = TfidfModel(bow_corpus)
    bow_counts = corpus2csc(tfidf_model[bow_corpus], num_terms=len(dictionary))
    bow_counts_sklearn_statsmodels = bow_counts.T  # Transpose for sklearn/statsmodels compatibility
    if clip and issparse(bow_counts_sklearn_statsmodels):
        bow_counts_sklearn_statsmodels.data = np.clip(bow_counts_sklearn_statsmodels.data, MIN_BOW_VALUE, MAX_BOW_VALUE)
    return bow_counts_sklearn_statsmodels


def reduce_bow_dimensions(matrix, n_components: int = 20, random_state: int = 42) -> np.ndarray:
    """Reduce a sparse TF-IDF BoW matrix to dense latent components via LSA (TruncatedSVD).

    This reduces the feature space from potentially thousands of BoW terms to a small
    number of latent semantic dimensions, enabling meaningful statsmodels MLE regression
    (reliable p-values, confidence intervals) when the sample count would otherwise be
    insufficient relative to the number of features.

    Args:
        matrix: Sparse TF-IDF matrix (shape: [n_samples, n_terms])
        n_components: Number of latent components to keep (default: 20)
        random_state: Random seed for reproducibility

    Returns:
        numpy array: Dense matrix of shape [n_samples, n_components]
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    return svd.fit_transform(matrix)


def text_processing_pipeline(
    texts: list[str], clip: bool = False, prune_dict: bool = True, reduce_bow_n_components: int = 20
) -> tuple[corpora.Dictionary, list[list[tuple[int, int]]]]:
    """Complete text processing pipeline: preprocess, stem, create dictionary, and convert to BoW"""
    preprocessed_texts = [preprocess_text(text) for text in texts]
    stemmed_texts = [stem_text(text) for text in preprocessed_texts]
    dictionary = create_dictionary(stemmed_texts)
    if prune_dict:
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
    bow_corpus = texts_to_bow(stemmed_texts, dictionary)
    matrix = texts_to_sparse_matrix(bow_corpus, dictionary, clip=clip)
    if reduce_bow_n_components:
        matrix = reduce_bow_dimensions(matrix, n_components=20)

    return dictionary, bow_corpus, matrix
