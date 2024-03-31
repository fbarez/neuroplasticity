from transformers import AutoModelForTokenClassification
from src import NEURONS_PER_LAYER, tokenizer
import torch


def prune_model(model_path: str, model_trainer, neurons_to_ablate):
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        id2label=model_trainer.id2label,
        label2id=model_trainer.label2id,
    )
    for neuron_pos in neurons_to_ablate:
        layer_id, neuron_index = divmod(neuron_pos, NEURONS_PER_LAYER)

        # FOR GPT2
        # weights = model.transformer.h[layer_id - 1].ln_2.weight.data

        # FOR DISTILBERT
        weights = model.distilbert.transformer.layer[
            layer_id - 1
        ].output_layer_norm.weight.data

        # # Prune the specified neuron by setting its weight to zero
        weights[neuron_index] = torch.zeros_like(weights[neuron_index])
        weights.requires_grad = False

    return model
