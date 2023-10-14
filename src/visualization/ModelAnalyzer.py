import os
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.interpretation.utils as utils
import neurox.interpretation.probeless as probeless
import neurox.analysis.corpus as corpus
from src import TOKENS_INPUT_PATH, TOKENS_LABEL_PATH, MODEL_CHECKPOINT, CONCEPT_LABEL


class ModelAnalyzer:
    def __init__(self, model_path, activations_path) -> None:
        self.activations = None
        self.tokens = None
        self.X = None
        self.y = None
        self.idx2label = None
        self.label2idx = None
        self.load_activations(model_path, activations_path)

    def load_activations(self, model_path, activations_path):
        model_path_type = model_path + "," + MODEL_CHECKPOINT
        if not os.path.exists(activations_path):
            transformers_extractor.extract_representations(
                model_path_type, TOKENS_INPUT_PATH, activations_path, aggregation="average"
            )
        self.activations, _ = data_loader.load_activations(activations_path)

    def load_tokens(self):
        self.tokens = data_loader.load_data(
            TOKENS_INPUT_PATH, TOKENS_LABEL_PATH, self.activations, 512
        )
        self.X, self.y, mapping = utils.create_tensors(
            self.tokens, self.activations, "NN"
        )
        self.label2idx, self.idx2label, _, _ = mapping

    def identify_concept_neurons(self):
        if self.tokens is None:
            self.load_tokens()
        top_neurons = probeless.get_neuron_ordering_for_tag(
            self.X, self.y, self.label2idx, CONCEPT_LABEL
        )
        # top_neurons, _ = probeless.get_neuron_ordering_for_all_tags(
        #     self.X, self.y, self.idx2label
        # )
        return top_neurons

    def show_top_words(self, concept_neurons):
        if self.tokens is None:
            self.load_tokens()
        top_words = {}
        for neuron_idx in concept_neurons:
            words = corpus.get_top_words(
                self.tokens, self.activations, neuron_idx, 5)
            top_words[neuron_idx] = words
            print(f"== {neuron_idx} ==")
            print(words)
        return top_words
