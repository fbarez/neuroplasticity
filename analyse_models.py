from build_models import RETRAIN_INCR_1, RETRAIN_INCR_2, RETRAIN_INCR_3, RETRAIN_INCR_4
from src import basic_model_path, basic_activations_path
from src.visualization.analyse_model import analyse_model
import pandas as pd

ACTIVATIONS_INCR_1 = "data/interim/retrained_name_activations_1.json"
ACTIVATIONS_INCR_2 = "data/interim/retrained_name_activations_2.json"
ACTIVATIONS_INCR_3 = "data/interim/retrained_name_activations_3.json"
ACTIVATIONS_INCR_4 = "data/interim/retrained_name_activations_4.json"

def compare_tokens(model_path, activations_path, analyser, analyser_1, analyser_2, analyser_3, analyser_4):
    # Extract top activating neurons for a model
    concept_neurons = analyser.identify_concept_neurons()
    # Compare to neurons in other models
    top_words = analyser.show_top_words(concept_neurons)
    top_words_1 = analyser_1.show_top_words(concept_neurons)
    top_words_2 = analyser_2.show_top_words(concept_neurons)
    top_words_3 = analyser_3.show_top_words(concept_neurons)
    top_words_4 = analyser_4.show_top_words(concept_neurons)
    df = pd.DataFrame(top_words.items(), columns=['neuron-id', 'current_concepts'])
    df['concepts_1'] = top_words_1.values()
    df['concepts_2'] = top_words_2.values()
    df['concepts_3'] = top_words_3.values()
    df['concepts_4'] = top_words_4.values()
    df.to_csv(f'data/processed/{model_path}_concepts.csv')

def main():
    base_analyser = analyse_model(basic_model_path, basic_activations_path)
    retrain_1_analyser = analyse_model(RETRAIN_INCR_1, ACTIVATIONS_INCR_1)
    retrain_2_analyser = analyse_model(RETRAIN_INCR_2, ACTIVATIONS_INCR_2)
    retrain_3_analyser = analyse_model(RETRAIN_INCR_3, ACTIVATIONS_INCR_3)
    retrain_4_analyser = analyse_model(RETRAIN_INCR_4, ACTIVATIONS_INCR_4)
    # Extract top activating neurons for models
    compare_tokens(basic_model_path, basic_activations_path, base_analyser, retrain_1_analyser, retrain_2_analyser, retrain_3_analyser, retrain_4_analyser)
    compare_tokens(RETRAIN_INCR_1, ACTIVATIONS_INCR_1, retrain_1_analyser, base_analyser, retrain_2_analyser, retrain_3_analyser, retrain_4_analyser)
    compare_tokens(RETRAIN_INCR_2, ACTIVATIONS_INCR_2, retrain_2_analyser, base_analyser, retrain_1_analyser, retrain_3_analyser, retrain_4_analyser)
    compare_tokens(RETRAIN_INCR_3, ACTIVATIONS_INCR_3, retrain_3_analyser, base_analyser, retrain_1_analyser, retrain_2_analyser, retrain_4_analyser)
    compare_tokens(RETRAIN_INCR_4, ACTIVATIONS_INCR_4, retrain_4_analyser, base_analyser, retrain_1_analyser, retrain_2_analyser, retrain_3_analyser)
    
if __name__ == '__main__':
    main()
