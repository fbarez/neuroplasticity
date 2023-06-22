import gensim.downloader as api
from gensim.models.fasttext import FastText
import pandas as pd
import ast

from src import PROCESSED_1, PROCESSED_2, PROCESSED_3, PROCESSED_4
from src import SIMILARITY_1, SIMILARITY_2, SIMILARITY_3, SIMILARITY_4

def extract_pre_words(row):
    string_data = row["current_concepts"]
    pre_words = [word for word, _ in ast.literal_eval(string_data)]
    return pre_words

def extract_post_words(row, col_name):
    string_data = row[col_name]
    post_words = [word for word, _ in ast.literal_eval(string_data)]
    return post_words

def compare_similarity(dataset_path):
    df = pd.read_csv(dataset_path)
    new_df = pd.DataFrame()
    new_df["neuron-id"] = df["neuron-id"]
    new_df["pre_words"] = df["current_concepts"]
    new_df["post_words"] = df["concepts_1"]
    for index, row in df.iterrows():
        pre_words = extract_pre_words(row)
        post_words = extract_post_words(row, "concepts_1")
        new_df.at[index, "pre_words"] = pre_words
        new_df.at[index, "post_words"] = post_words
        new_df.at[index, "similarity"] = conceptnet.wv.n_similarity(pre_words, post_words)
    return new_df

if __name__ == '__main__':

    try:
        conceptnet = FastText.load("fasttext.model")
    except:
        corpus = api.load('text8')
        conceptnet = FastText(corpus)
        conceptnet.save("fasttext.model")

    retrained_1_df = compare_similarity(PROCESSED_1)
    retrained_1_df.to_csv(SIMILARITY_1)

    retrained_2_df = compare_similarity(PROCESSED_2)
    retrained_2_df.to_csv(SIMILARITY_2)

    retrained_3_df = compare_similarity(PROCESSED_3)
    retrained_3_df.to_csv(SIMILARITY_3)

    retrained_4_df = compare_similarity(PROCESSED_4)
    retrained_4_df.to_csv(SIMILARITY_4)
