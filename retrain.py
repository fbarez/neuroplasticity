import pandas as pd
from src import pruned_model_path, retrained_model_path, retrained_activations_path

from src.models.train_model import train_model
from src.models.evaluate_model import evaluate
from src.visualization.visualize import get_retrained_model
from src.visualization.analyse_model import analyse_model

# To run: python3 retrain.py

def retrain_pruned(model_trainer):
    # Retrain the pruned model
    retrained_model = get_retrained_model(
        retrained_model_path, pruned_model_path, model_trainer)
    # Examine the retrained model for concepts and compare with old model
    retrained_analyser = analyse_model(
        retrained_model_path, retrained_activations_path)
    return retrained_analyser


def main():
    print("Running retraining experiment...")
    model_trainer = train_model()
    retrained_analyser = retrain_pruned(model_trainer)

    # Perform analysis
    new_concept_neurons = retrained_analyser.identify_concept_neurons()
    post_top_words = retrained_analyser.show_top_words(new_concept_neurons)
    pre_top_words = basic_analyser.show_top_words(new_concept_neurons)
    plastic_df = pd.DataFrame(data=[pre_top_words.items(), post_top_words.items(
    )], columns=['neuron_id', 'pre_top_words', 'post_top_words'])
    plastic_df.to_csv('data/processed/locations_plastic.csv')


if __name__ == '__main__':
    main()
