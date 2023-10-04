from src import BASIC_MODEL_PATH, BASIC_ACTIVATIONS_PATH, PRUNED_MODEL_PATH, NEURONS_PER_LAYER, NUM_LAYERS
from src.models.ModelTrainer import ModelTrainer
from src.features.build_features import load_token_class_dataset
from src.visualization.get_models import get_basic_model, get_incr_retrained_model
from src.models.prune_model import prune_model
from src.visualization.ModelAnalyzer import ModelAnalyzer
from src.models.evaluate_model import evaluate
import pandas as pd

RETRAIN_INCR_1_PATH = "models/retrained_model_1"
RETRAIN_INCR_2_PATH = "models/retrained_model_2"
RETRAIN_INCR_3_PATH = "models/retrained_model_3"
RETRAIN_INCR_4_PATH = "models/retrained_model_4"

ACTIVATIONS_INCR_1_PATH = "data/interim/retrained_activations_1.json"
ACTIVATIONS_INCR_2_PATH = "data/interim/retrained_activations_2.json"
ACTIVATIONS_INCR_3_PATH = "data/interim/retrained_activations_3.json"
ACTIVATIONS_INCR_4_PATH = "data/interim/retrained_activations_4.json"

def build_base_model(model_trainer, dataset):
    basic_model = get_basic_model(model_trainer)
    evaluate(basic_model, dataset["validation"])

def build_pruned_model(model_trainer, dataset):
    # Identify neurons in the basic model to ablate
    basic_analyser = ModelAnalyzer(BASIC_MODEL_PATH, BASIC_ACTIVATIONS_PATH)
    neurons_to_prune = basic_analyser.identify_concept_neurons()
    num_prune = (NEURONS_PER_LAYER * NUM_LAYERS) // 2
    # Neurons to prune are sorted by weight in ascending order. Prune most important from end of list.
    pruned_model = prune_model(BASIC_MODEL_PATH, model_trainer, neurons_to_prune[-num_prune:])
    pruned_model.save_pretrained(PRUNED_MODEL_PATH)
    evaluate(pruned_model, dataset["validation"])

def incr_retrain_model(model_trainer, dataset, pre_model_path, post_model_path):
    incr_model = get_incr_retrained_model(post_model_path, pre_model_path, model_trainer)
    evaluate(incr_model, dataset["validation"])

def build_models():
    # Load dataset and trainer
    dataset = load_token_class_dataset()
    model_trainer = ModelTrainer()
    # Train models
    build_base_model(model_trainer, dataset)
    build_pruned_model(model_trainer, dataset)
    incr_retrain_model(model_trainer, dataset, PRUNED_MODEL_PATH, RETRAIN_INCR_1_PATH)
    incr_retrain_model(model_trainer, dataset, RETRAIN_INCR_1_PATH, RETRAIN_INCR_2_PATH)
    incr_retrain_model(model_trainer, dataset, RETRAIN_INCR_2_PATH, RETRAIN_INCR_3_PATH)
    incr_retrain_model(model_trainer, dataset, RETRAIN_INCR_3_PATH, RETRAIN_INCR_4_PATH)

def compare_tokens(model_path, analyser, analyser_1, analyser_2, analyser_3, analyser_4):
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

def analyse_models():
    base_analyser = ModelAnalyzer(BASIC_MODEL_PATH, BASIC_MODEL_PATH)
    retrain_1_analyser = ModelAnalyzer(RETRAIN_INCR_1_PATH, ACTIVATIONS_INCR_1_PATH)
    retrain_2_analyser = ModelAnalyzer(RETRAIN_INCR_2_PATH, ACTIVATIONS_INCR_2_PATH)
    retrain_3_analyser = ModelAnalyzer(RETRAIN_INCR_3_PATH, ACTIVATIONS_INCR_3_PATH)
    retrain_4_analyser = ModelAnalyzer(RETRAIN_INCR_4_PATH, ACTIVATIONS_INCR_4_PATH)
    # Extract top activating neurons for models
    compare_tokens(BASIC_MODEL_PATH, base_analyser, retrain_1_analyser, retrain_2_analyser, retrain_3_analyser, retrain_4_analyser)
    compare_tokens(RETRAIN_INCR_1_PATH, retrain_1_analyser, base_analyser, retrain_2_analyser, retrain_3_analyser, retrain_4_analyser)
    compare_tokens(RETRAIN_INCR_2_PATH, retrain_2_analyser, base_analyser, retrain_1_analyser, retrain_3_analyser, retrain_4_analyser)
    compare_tokens(RETRAIN_INCR_3_PATH, retrain_3_analyser, base_analyser, retrain_1_analyser, retrain_2_analyser, retrain_4_analyser)
    compare_tokens(RETRAIN_INCR_4_PATH, retrain_4_analyser, base_analyser, retrain_1_analyser, retrain_2_analyser, retrain_3_analyser)
    
if __name__ == '__main__':
    build_models()
    analyse_models()
