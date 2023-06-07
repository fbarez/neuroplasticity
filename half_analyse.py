from src import basic_model_path, basic_activations_path, pruned_model_path
from src.visualization.analyse_model import analyse_model
from src import retrained_model_path, retrained_activations_path
import pandas as pd

def half_analyse():
    analyser = analyse_model(retrained_model_path, retrained_activations_path)
    basic_analyser = analyse_model(basic_model_path, basic_activations_path)
    # Extract top activating neurons for a model
    concept_neurons = analyser.identify_concept_neurons()
    top_words = analyser.show_top_words(concept_neurons)
    top_words_basic = basic_analyser.show_top_words(concept_neurons)
    df = pd.DataFrame(top_words.items(), columns=['neuron-id', 'retrained_top'])
    df['retrained_base'] = top_words_basic.values()
    df.to_csv(f'data/processed/full_retrain_concepts.csv')

if __name__ == '__main__':
    half_analyse()