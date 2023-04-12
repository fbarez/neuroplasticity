from models.train_model import train_model
from features.build_features import load_token_class_dataset
from models.evaluate_model import evaluate
from models.prune_model import prune_model
from analyse_model import analyse_model
from src import (
    basic_model_path,
    pruned_model_path,
    retrained_model_path,
    basic_activations_path,
    retrained_activations_path,
)


def visualise():
    dataset = load_token_class_dataset()
    # Train and evaluate a basic model on pretrained DistilBert
    model_trainer = train_model()
    basic_model = model_trainer.train_basic_model()
    basic_model.save_pretrained(basic_model_path)
    evaluate(basic_model, dataset["validation"])
    # Identify neurons in the basic neuron to ablate
    basic_analyser = analyse_model(basic_model_path, basic_activations_path)
    neurons_to_prune = basic_analyser.identify_concept_neurons()
    basic_analyser.show_top_words(neurons_to_prune)
    # Ablate neurons
    pruned_model = prune_model(basic_model_path, model_trainer, neurons_to_prune)
    pruned_model.save_pretrained(pruned_model_path)
    evaluate(pruned_model, dataset["validation"])
    # Retrain the pruned model
    retrained_model = model_trainer.retrain_pruned_model(pruned_model_path)
    retrained_model.save_pretrained(retrained_model_path)
    # Examine the retrained model for concepts and compare with old model
    retrained_analyser = analyse_model(retrained_model_path, retrained_activations_path)
    new_concept_neurons = retrained_analyser.identify_concept_neurons()
    retrained_analyser.show_top_words(new_concept_neurons)
    basic_analyser.show_top_words(new_concept_neurons)
