from transformers import AutoTokenizer

model_max_neurons = 768
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
basic_model_path = "../../models/basic_model"
pruned_model_path = "../../models/pruned_model"
retrained_model_path = "../../models/retrained_model"
token_inputs_path = "../../data/processed/location.in"
token_labels_path = "../../data/processed/location.label"
basic_activations_path = "../../data/processed/basic_activations.json"
retrained_activations_path = "../../data/processed/retrained_activations.json"
