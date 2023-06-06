from src import basic_model_path, basic_activations_path, pruned_model_path
from src.models.train_model import train_model
from src.features.build_features import load_token_class_dataset
from src.visualization.visualize import get_basic_model, get_retrained_model
from src.models.prune_model import prune_model
from src.visualization.analyse_model import analyse_model
from src.models.evaluate_model import evaluate
from src import retrained_model_path, retrained_activations_path
import pandas as pd

def build_base_model(model_trainer, dataset):
    basic_model = get_basic_model(model_trainer)
    evaluate(basic_model, dataset["validation"])

def build_pruned_model(model_trainer, dataset):
    # Identify neurons in the basic model to ablate
    basic_analyser = analyse_model(basic_model_path, basic_activations_path)
    neurons_to_prune = basic_analyser.identify_concept_neurons()
    print(len(neurons_to_prune))
    pruned_model = prune_model(basic_model_path, model_trainer, neurons_to_prune)
    pruned_model.save_pretrained(pruned_model_path)
    evaluate(pruned_model, dataset["validation"])

def full_retrain():
    # Load dataset and trainer
    dataset = load_token_class_dataset()
    model_trainer = train_model()
    # Retrain pruned model
    build_base_model(model_trainer, dataset)
    build_pruned_model(model_trainer, dataset)
    retrained_model = get_retrained_model(retrained_model_path, pruned_model_path, model_trainer)
    # Analyse retrained model
    evaluate(retrained_model, dataset["validation"])
    # analyser = analyse_model(retrained_model_path, retrained_activations_path)
    # basic_analyser = analyse_model(basic_model_path, basic_activations_path)
    # # Extract top activating neurons for a model
    # concept_neurons = analyser.identify_concept_neurons()
    # top_words = analyser.show_top_words(concept_neurons)
    # top_words_basic = basic_analyser.show_top_words(concept_neurons)
    # df = pd.DataFrame(top_words.items(), columns=['neuron-id', 'retrained_top'])
    # df['retrained_base'] = top_words_basic.values()
    # df.to_csv(f'data/processed/full_retrain_concepts.csv')

if __name__ == '__main__':
    full_retrain()