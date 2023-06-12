import gensim.downloader
import pandas as pd
import ast

PROCESSED_1 = "data/processed/retrained_name_model_1_concepts.csv"
PROCESSED_2 = "data/processed/retrained_name_model_2_concepts.csv"
PROCESSED_3 = "data/processed/retrained_name_model_3_concepts.csv"
PROCESSED_4 = "data/processed/retrained_name_model_4_concepts.csv"

def extract_pre_words(row, pre_words_col):
    string_data = row[pre_words_col]
    pre_words = [word for word, _ in ast.literal_eval(string_data)]
    return pre_words

def extract_post_words(row, post_words_col):
    string_data = row[post_words_col]
    post_words = [word for word, _ in ast.literal_eval(string_data)]
    return post_words

def vectorise(words, conceptnet):
    """ Return set of word embeddings for given list of words. """
    embeddings = []
    for word in words:
        if conceptnet.wv.has_index_for(word):
            embeddings.append(model.wv[word])
    return set(embeddings)

def compare_similarity(dataset_path, pre_words_col, post_words_col, num):
    conceptnet = gensim.downloader.load('conceptnet-numberbatch-17-06-300')
    df = pd.read_csv(dataset_path)
    for index, row in df.iterrows():
        if index > 5: break
        pre_words = extract_pre_words(row, pre_words_col)
        print(pre_words)
        post_words = extract_post_words(row, post_words_col)
        print(post_words)
        df.at[index, "similarity"] = conceptnet.n_similarity(pre_words, post_words)
    print(df)
    df.to_csv(f'data/processed/similarity_{num}.csv')
    return df

if __name__ == '__main__':
    compare_similarity(PROCESSED_1, "current_concepts", "concepts_1", 1)
    # compare_similarity(PROCESSED_2, "current_concepts", "concepts_1", 2)
    # compare_similarity(PROCESSED_3, "current_concepts", "concepts_1", 3)
    # compare_similarity(PROCESSED_4, "current_concepts", "concepts_1", 4)
