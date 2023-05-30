from transformers import AutoTokenizer

model_max_neurons = 768
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
basic_model_path = "models/basic_model"
pruned_model_path = "models/pruned_model"
retrained_model_path = "models/retrained_model"
token_inputs_path = "data/interim/location_tokens.txt"
token_labels_path = "data/interim/location_labels.txt"
basic_activations_path = "data/interim/basic_activations.json"
retrained_activations_path = "data/interim/retrained_activations.json"
