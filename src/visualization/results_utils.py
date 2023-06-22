import numpy as np
import pandas as pd
import seaborn as sns

import ast
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import scipy.stats as st
import math


def measure_concept_relevance(row_index, total_neurons):
    # Normalised measurement of saliency ranking out of total number of neurons
    min_rank = total_neurons / 100
    max_rank = 1
    percentile, _ = divmod(row_index, 100)
    rank = percentile + 1
    normalized_rank = (rank - min_rank) / (max_rank - min_rank)
    return abs(normalized_rank)


def has_word_starting_with_backslash(word_list):
    # Ignore words with '\x80' (NULL)
    for word in word_list:
        if not word.isalpha():
            return True
    return False


def build_heatmap_data(p, total_neurons):
    # Convert into grid of 7 * 768 neurons with saliency values
    heat_data = [[float('NaN')] * 768 for _ in range(6)]
    heatdf = pd.DataFrame(heat_data)
    for index, row in p.iterrows():
        nid = row['neuron-id']
        layer_id, neuron_index = divmod(nid, 768)
        string_data = row["current_concepts"]
        words = [word for word, _ in ast.literal_eval(string_data)]
        if not has_word_starting_with_backslash(words):
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


def build_sim_data(df):
    heat_data = [[float('NaN')] * 768 for _ in range(6)]
    heatdf = pd.DataFrame(heat_data)
    for index, row in df.iterrows():
        nid = row['neuron-id']
        layer_id, neuron_index = divmod(nid, 768)
        # If nothing changed, exclude
        pre_words = set(ast.literal_eval(row['pre_words']))
        post_words = set(ast.literal_eval(row['post_words']))
        if pre_words != post_words:
            heatdf.loc[layer_id, neuron_index] = row['similarity']
    return heatdf


def build_sim_heatmap_figure(df, name, save=False):
    # Set the width and height of the figure
    plt.figure(figsize=(30, 7))

    # Heatmap showing average arrival delay for each airline by month
    cmap = mpl.colormaps['seismic']
    cmap.set_bad('black')
    sns.heatmap(data=df, vmin=-1, vmax=1, cmap=cmap)

    plt.xlabel("Neuron index")
    plt.ylabel("Layer")

    if save:
        plt.savefig(f'{name}.pdf')


def build_zoom_sim_heatmap_figure(df, name, save=False):
    # Set the width and height of the figure
    plt.figure(figsize=(5, 5))

    # Slice the dataframe to show
    data = df.iloc[:, 360:391]

    # Heatmap showing average arrival delay for each airline by month
    cmap = mpl.colormaps['seismic']
    cmap.set_bad('black')
    sns.heatmap(data=data, vmin=-1, vmax=1, cmap=cmap)

    plt.xlabel("Neuron index")
    plt.ylabel("Layer")

    if save:
        plt.savefig(f'{name}.pdf')


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
    for index, row in heatdf.iterrows():
        row_clean = [x for x in row.tolist() if not math.isnan(x)]
        average = np.mean(row_clean)
        (low, high) = st.norm.interval(confidence=0.95,
                                       loc=np.mean(row_clean), scale=st.sem(row_clean))
        error = abs(high - low) / 2
        df.loc[len(df)] = [len(df), average, error]
    return df


def get_random_hats(df, retrain_4_heatdf, layer):
    layer_df = df.loc[(df['neuron-id'] >= (768 * layer)) &
                      (df['neuron-id'] <= (768 * (layer + 1)))]
    sample = layer_df.sample(n=4, random_state=151)
    saliencies = []
    for index, row in sample.iterrows():
        nid = row['neuron-id']
        print(nid)
        layer_id, neuron_index = divmod(nid, 768)
        s = retrain_4_heatdf.iloc[layer_id][neuron_index]
        if not math.isnan(s):
            saliencies.append(s)
    print(saliencies)
    print(np.mean(saliencies))
    return sample["HAT"].tolist()
