from src import BASE_CONCEPT_PATH, NEURONS_PER_LAYER, NUM_LAYERS
import gensim.downloader as api
from gensim.models.fasttext import FastText
import numpy as np
import pandas as pd
import seaborn as sns
import ast
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as st
import math


def get_not_pruned():
    """ Calculate indexes of neurons which were not pruned """
    base_df = pd.read_csv(BASE_CONCEPT_PATH)
    not_pruned_num = len(base_df) // 2
    return base_df['neuron-id'].tail(not_pruned_num).tolist()


def measure_concept_relevance(row_index, total_neurons):
    """ Normalised measurement of saliency ranking out of total number of neurons """
    min_rank = total_neurons / 100
    max_rank = 1
    percentile, _ = divmod(row_index, 100)
    rank = percentile + 1
    normalized_rank = (rank - min_rank) / (max_rank - min_rank)
    return abs(normalized_rank)


def null_words(word_list):
    # Ignore words with mostly '\x80' (NULL)
    for word in word_list:
        if word.isalpha():
            return False
    return True


def build_heatmap_data(p, total_neurons):
    # Convert into grid of 7 * 768 neurons with saliency values
    heat_data = [[float('NaN')] * NEURONS_PER_LAYER for _ in range(NUM_LAYERS)]
    heatdf = pd.DataFrame(heat_data)
    for index, row in p.iterrows():
        nid = row['neuron-id']
        layer_id, neuron_index = divmod(nid, NEURONS_PER_LAYER)
        string_data = row["current_concepts"]
        words = [word for word, _ in ast.literal_eval(string_data)]
        if not null_words(words):
            heatdf.loc[layer_id, neuron_index] = measure_concept_relevance(
                index, total_neurons)
    return heatdf


def build_heatmap_figure(df, name, save=False):
    # Set the width and height of the figure
    plt.figure(figsize=(30, 10))

    # Heatmap showing average arrival delay for each airline by month
    cmap = mpl.colormaps['coolwarm']
    cmap.set_bad('black')
    sns.heatmap(data=df, cmap=cmap)

    plt.xlabel("Neuron index")
    plt.ylabel("Layer")

    if save:
        plt.savefig(f'{name}.pdf', bbox_inches='tight')


def build_zoom_heatmap_figure(df, name, save=False):
    # Set the width and height of the figure
    plt.figure(figsize=(5, 5))

    # Slice the dataframe to show
    data = df.iloc[:, 360:391]

    # Heatmap showing average arrival delay for each airline by month
    cmap = mpl.colormaps['coolwarm']
    cmap.set_bad('black')
    sns.heatmap(data=data, cmap=cmap, vmin=0, vmax=1)

    plt.xlabel("Neuron index")
    plt.ylabel("Layer")

    if save:
        plt.savefig(f'{name}.pdf')


def filter_ablated(df):
    filtered_df = pd.DataFrame(columns=['neuron-id', 'current_concepts'])
    for _, row in df.iterrows():
        nid = row['neuron-id']
        if nid in get_not_pruned():
            filtered_df.loc[len(filtered_df)] = row
    return filtered_df


def load_word_embeddings_model():
    try:
        conceptnet = FastText.load("fasttext.model")
    except:
        corpus = api.load('text8')
        conceptnet = FastText(corpus)
        conceptnet.save("fasttext.model")
    return conceptnet


def extract_pre_words(row):
    string_data = row["current_concepts"]
    pre_words = [word for word, _ in ast.literal_eval(string_data)]
    return pre_words


def extract_post_words(row, col_name):
    string_data = row[col_name]
    post_words = [word for word, _ in ast.literal_eval(string_data)]
    return post_words


def compare_similarity(dataset_path, col_name):
    df = pd.read_csv(dataset_path)
    new_df = pd.DataFrame()
    new_df["neuron-id"] = df["neuron-id"]
    new_df["pre_words"] = df["current_concepts"]
    new_df["post_words"] = df[col_name]
    conceptnet = load_word_embeddings_model()
    for index, row in df.iterrows():
        pre_words = extract_pre_words(row)
        post_words = extract_post_words(row, col_name)
        new_df.at[index, "pre_words"] = pre_words
        new_df.at[index, "post_words"] = post_words
        new_df.at[index, "similarity"] = conceptnet.wv.n_similarity(
            pre_words, post_words)
    return new_df


def build_sim_data(df):
    heat_data = [[float('NaN')] * NEURONS_PER_LAYER for _ in range(NUM_LAYERS)]
    heatdf = pd.DataFrame(heat_data)
    for _, row in df.iterrows():
        nid = row['neuron-id']
        layer_id, neuron_index = divmod(nid, NEURONS_PER_LAYER)
        # If nothing changed, exclude
        pre_words = set(ast.literal_eval(row['pre_words']))
        post_words = set(ast.literal_eval(row['post_words']))
        if pre_words != post_words:
            heatdf.loc[layer_id, neuron_index] = row['similarity']
    return heatdf


def mean_saliency(heatdf):
    df = pd.DataFrame(columns=['layer', 'mean_saliency', 'error'])
    for index, row in heatdf.iterrows():
        row_clean = [x for x in row.tolist() if not math.isnan(x)]
        average = np.mean(row_clean)
        (low, high) = st.norm.interval(confidence=0.95,
                                       loc=np.mean(row_clean), scale=st.sem(row_clean))
        error = abs(high - low) / 2
        df.loc[len(df)] = [len(df), average, error]
    return df


def mean_similarity(heatdf):
    df = pd.DataFrame(columns=['layer', 'mean_similarity', 'error'])
    for _, row in heatdf.iterrows():
        row_clean = [x for x in row.tolist() if not math.isnan(x)]
        average = np.mean(row_clean)
        (low, high) = st.norm.interval(confidence=0.95,
                                       loc=np.mean(row_clean), scale=st.sem(row_clean))
        error = abs(high - low) / 2
        df.loc[len(df)] = [len(df), average, error]
    return df


def get_random_hats(df, saliency_df, layer):
    layer_df = df.loc[(df['neuron-id'] >= (NEURONS_PER_LAYER * layer))
                      & (df['neuron-id'] <= (NEURONS_PER_LAYER * (layer + 1)))]
    sample = layer_df.sample(n=4, random_state=151)
    saliencies = []
    for _, row in sample.iterrows():
        nid = row['neuron-id']
        layer_id, neuron_index = divmod(nid, NEURONS_PER_LAYER)
        s = saliency_df.iloc[layer_id][neuron_index]
        if not math.isnan(s):
            saliencies.append(s)
    print("mean saliency", np.mean(saliencies))
    return sample["HAT"].tolist()
