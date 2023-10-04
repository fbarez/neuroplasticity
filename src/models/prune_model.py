from transformers import AutoModelForTokenClassification
from src import NEURONS_PER_LAYER
from src.models.ModelTrainer import train_model
import torch


def prune_model(model_path: str, model_trainer: train_model, neurons_to_ablate):
    pruned_model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        id2label=model_trainer.id2label,
        label2id=model_trainer.label2id,
    )
    for neuron_pos in neurons_to_ablate:
        layer_id, neuron_index = divmod(neuron_pos, NEURONS_PER_LAYER)
        # Access the layer's weights
        weights = pruned_model.distilbert.transformer.layer[
            layer_id - 1
        ].output_layer_norm.weight.data
        # Prune the specified neuron by setting its weight to zero
        weights[neuron_index] = torch.zeros_like(weights[neuron_index])
        # Freeze the weights such that they are not updated during retraining
        weights.requires_grad = False

    return pruned_model
