from src import location_remap_path
import gensim.downloader
import pandas as pd
import ast

def extract_pre_words(row):
    string_data = row["pre_top_words"]
    pre_words = [word for word, _ in ast.literal_eval(string_data)]
    return pre_words

def extract_post_words(df):
    string_data = row["post_top_words"]
    post_words = [word for word, _ in ast.literal_eval(string_data)]
    return post_words

def vectorise(words, conceptnet):
    """ Return set of word embeddings for given list of words. """
    embeddings = []
    for word in words:
        if conceptnet.wv.has_index_for(word):
            embeddings.append(model.wv[word])
    return set(embeddings)

def compare_similarity(dataset_path):
    conceptnet = gensim.downloader.load('conceptnet-numberbatch-17-06-300')
    df = pd.read_csv(dataset_path)
    for index, row in df.iterrows():
        pre_words = extract_pre_words(row)
        post_words = extract_post_words(row)
        df.at[index, "similarity"] = conceptnet.n_similarity(pre_words, post_words)
    print(df)
    return df

if __name__ == '__main__':
    compare_similarity(location_remap_path)
