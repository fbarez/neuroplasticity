import pandas as pd
import os
from src import basic_model_path, basic_activations_path, pruned_model_path, retrained_model_path, retrained_activations_path

from src.models.train_model import train_model
from src.features.build_features import load_token_class_dataset
from src.models.evaluate_model import evaluate
from src.visualization.visualize import get_basic_model, get_retrained_model
from src.visualization.analyse_model import analyse_model
from src.models.prune_model import prune_model
from src.models.evaluate_model import evaluate

# To run: python3 main.py

def train_basic(model_trainer, dataset):
    # Train and evaluate a basic model on pretrained DistilBert
    basic_model = get_basic_model(model_trainer)
    evaluate(basic_model, dataset["validation"])


def find_neurons_to_prune():
    # Identify neurons in the basic model to ablate
    basic_analyser = analyse_model(basic_model_path, basic_activations_path)
    neurons_to_prune = basic_analyser.identify_concept_neurons()
    basic_top_words = basic_analyser.show_top_words(neurons_to_prune)
    prune_df = pd.DataFrame(data=basic_top_words.items(), columns=[
                            'neuron_id', 'top_words'])
    prune_df.to_csv('data/processed/locations_prune.csv')
    return basic_analyser, neurons_to_prune


def prune(model_trainer, neurons_to_prune, dataset):
    # Ablate neurons
    pruned_model = prune_model(
        basic_model_path, model_trainer, neurons_to_prune)
    pruned_model.save_pretrained(pruned_model_path)
    evaluate(pruned_model, dataset["validation"])


def retrain_pruned(model_trainer):
    # Retrain the pruned model
    retrained_model = get_retrained_model(
        retrained_model_path, pruned_model_path, model_trainer)
    # Examine the retrained model for concepts and compare with old model
    retrained_analyser = analyse_model(
        retrained_model_path, retrained_activations_path)
    return retrained_analyser


def main():
    print("Running remapping experiment...")
    # print(os.path.exists(basic_model_path))
    # print(os.path.exists("data/interim/location_tokens.txt"))
    # print(os.path.exists(basic_activations_path))

    dataset = load_token_class_dataset()
    model_trainer = train_model()
    train_basic(model_trainer, dataset)
    basic_analyser, neurons_to_prune = find_neurons_to_prune()
    prune(model_trainer, neurons_to_prune, dataset)
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
