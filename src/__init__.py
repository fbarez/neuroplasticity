from transformers import AutoTokenizer

model_max_neurons = 768
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
basic_model_path = "models/basic_model_name"
pruned_model_path = "models/pruned_model_name"
retrained_model_path = "models/retrained_model_name"
token_inputs_path = "data/interim/name_tokens.txt"
token_labels_path = "data/interim/name_labels.txt"
basic_activations_path = "data/interim/basic_name_activations.json"
retrained_activations_path = "data/interim/retrained_name_activations.json"
