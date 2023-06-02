from src import location_remap_path
import gensim.downloader

def extract_pre_words():
    pass

def extract_post_words():
    pass

def vectorise(words, conceptnet):
    """ Return set of word embeddings for given list of words. """
    embeddings = []
    for word in words:
        if conceptnet.wv.has_index_for(word):
            embeddings.append(model.wv[word])
    return set(embeddings)

def compare_similarity():
    conceptnet = gensim.downloader.load('conceptnet-numberbatch-17-06-300')
    pre_words = extract_pre_words()
    post_words = extract_post_words()
    return conceptnet.n_similarity(pre_words, post_words)
