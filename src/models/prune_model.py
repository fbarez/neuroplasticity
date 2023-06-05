from transformers import AutoModelForTokenClassification
from src import model_max_neurons
from src.models.train_model import train_model
import torch


def prune_model(model_path: str, model_trainer: train_model, neurons_to_ablate):
    pruned_model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        id2label=model_trainer.id2label,
        label2id=model_trainer.label2id,
    )
    for neuron_pos in neurons_to_ablate:
        layer_id, neuron_index = divmod(neuron_pos, model_max_neurons)
        # Access the layer's weights
        weights = pruned_model.distilbert.transformer.layer[
            layer_id - 1
        ].output_layer_norm.weight
        biases = pruned_model.distilbert.transformer.layer[
            layer_id - 1
        ].output_layer_norm.bias
        # Prune the specified neuron by setting its weight and bias to zero
        weights.data[neuron_index] = torch.zeros_like(
            weights.data[neuron_index])
        biases.data[neuron_index] = torch.zeros_like(
            biases.data[neuron_index])
        # Freeze the weights such that they are not updated during retraining
        # weights.requires_grad = False
        # biases.requires_grad = False

    return pruned_model


def get_neurons_weights(model_path: str, model_trainer: train_model, neurons):
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        id2label=model_trainer.id2label,
        label2id=train_model.label2id,
    )
    neurons_desc = {}
    for neuron_pos in neurons:
        layer_id, neuron_index = divmod(neuron_pos, model_max_neurons)
        # Access the layer's weights
        weights = model.distilbert.transformer.layer[
            layer_id - 1
        ].output_layer_norm.weight.data
        biases = model.distilbert.transformer.layer[
            layer_id - 1
        ].output_layer_norm.bias.data
        neurons_desc[neuron_pos] = (
            weights[neuron_index], biases[neuron_index])

    return neurons_desc
