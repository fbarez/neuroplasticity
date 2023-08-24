from src import basic_model_path, basic_activations_path, pruned_model_path
from src.models.train_model import train_model
from src.features.build_features import load_token_class_dataset
from src.visualization.visualize import get_basic_model, get_incr_retrained_model
from src.models.prune_model import prune_model
from src.visualization.analyse_model import analyse_model
from src.models.evaluate_model import evaluate

RETRAIN_INCR_1 = "models/retrained_model_1"
RETRAIN_INCR_2 = "models/retrained_model_2"
RETRAIN_INCR_3 = "models/retrained_model_3"
RETRAIN_INCR_4 = "models/retrained_model_4"

def build_base_model(model_trainer, dataset):
    basic_model = get_basic_model(model_trainer)
    evaluate(basic_model, dataset["validation"])

def build_pruned_model(model_trainer, dataset):
    # Identify neurons in the basic model to ablate
    basic_analyser = analyse_model(basic_model_path, basic_activations_path)
    neurons_to_prune = basic_analyser.identify_concept_neurons()
    pruned_model = prune_model(basic_model_path, model_trainer, neurons_to_prune)
    pruned_model.save_pretrained(pruned_model_path)
    evaluate(pruned_model, dataset["validation"])

def incr_retrain_model(model_trainer, dataset, pre_model_path, post_model_path):
    incr_model = get_incr_retrained_model(post_model_path, pre_model_path, model_trainer)
    evaluate(incr_model, dataset["validation"])

def build_models():
    # Load dataset and trainer
    dataset = load_token_class_dataset()
    model_trainer = train_model()
    # Train models
    build_base_model(model_trainer, dataset)
    build_pruned_model(model_trainer, dataset)
    incr_retrain_model(model_trainer, dataset, pruned_model_path, RETRAIN_INCR_1)
    incr_retrain_model(model_trainer, dataset, RETRAIN_INCR_1, RETRAIN_INCR_2)
    incr_retrain_model(model_trainer, dataset, RETRAIN_INCR_2, RETRAIN_INCR_3)
    incr_retrain_model(model_trainer, dataset, RETRAIN_INCR_3, RETRAIN_INCR_4)

if __name__ == '__main__':
    build_models()