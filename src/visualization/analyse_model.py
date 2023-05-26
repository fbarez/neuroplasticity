import os
import src.neurox_module.neurox.data.extraction.transformers_extractor as transformers_extractor
import src.neurox_module.neurox.data.loader as data_loader
import src.neurox_module.neurox.interpretation.utils as utils
import src.neurox_module.neurox.interpretation.linear_probe as linear_probe
import src.neurox_module.neurox.interpretation.clustering as clustering
import src.neurox_module.neurox.interpretation.probeless as probeless
import src.neurox_module.neurox.analysis.corpus as corpus
from collections import defaultdict
from src import token_inputs_path, token_labels_path, model_checkpoint


class analyse_model:
    def __init__(self, model_path, activations_path) -> None:
        self.activations = None
        self.tokens = None
        self.X = None
        self.y = None
        self.idx2label = None
        self.label2idx = None
        self.probe = None
        self.cluster_labels = None
        self.cluster_map = defaultdict(list)

        self.load_activations(model_path, activations_path)
        self.load_tokens()
        self.train_probe()

    def load_activations(self, model_path, activations_path):
        model_path_type = model_path + "," + model_checkpoint
        transformers_extractor.extract_representations(
            model_path_type, token_inputs_path, activations_path, aggregation="average"
        )
        self.activations, num_layers = data_loader.load_activations(activations_path)

    def load_tokens(self):
        self.tokens = data_loader.load_data(
            token_inputs_path, token_labels_path, self.activations, 512
        )
        self.X, self.y, mapping = utils.create_tensors(
            self.tokens, self.activations, "NN"
        )
        self.label2idx, self.idx2label, src2idx, idx2src = mapping

    def train_probe(self):
        self.probe = linear_probe.train_logistic_regression_probe(self.X, self.y)
        # Evaluate probe metrics
        scores = linear_probe.evaluate_probe(
            self.probe, self.X, self.y, idx_to_class=self.idx2label
        )
        print(scores)

    def identify_clusters(self):
        self.cluster_labels = clustering.create_correlation_clusters(self.X)
        # Convert cluster labels list to dictionary
        for idx in range(len(self.cluster_labels)):
            label = self.cluster_labels[idx]
            self.cluster_map[label].append(idx)

    def identify_concept_neurons(self):
        self.identify_clusters()
        top_neurons = probeless.get_neuron_ordering_for_tag(
            self.X, self.y, self.label2idx, "SEM:named_entity:location"
        )

        # Identify clusters containing concept neurons
        concept_clusters = []
        for neuron_idx in top_neurons:
            if self.cluster_labels[neuron_idx] - 1 not in concept_clusters:
                concept_clusters.append(self.cluster_labels[neuron_idx] - 1)

        # get all neurons associated with identified clusters
        associated_neurons = []
        for cluster_idx in concept_clusters:
            associated_neurons += self.cluster_map[cluster_idx]
        print(associated_neurons)

        return associated_neurons

    def show_top_words(self, concept_neurons):
        for neuron_idx in concept_neurons:
            print(
                neuron_idx,
                corpus.get_top_words(self.tokens, self.activations, neuron_idx),
            )
