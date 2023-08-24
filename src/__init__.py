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

BASE_CONCEPT_PATH = "data/processed/basic_model_concepts.csv"
PROCESSED_1 = "data/processed/retrained_model_1_concepts.csv"
PROCESSED_2 = "data/processed/retrained_model_2_concepts.csv"
PROCESSED_3 = "data/processed/retrained_model_3_concepts.csv"
PROCESSED_4 = "data/processed/retrained_model_4_concepts.csv"

SIMILARITY_1 = "data/processed/retrained_1_model_similarity.csv"
SIMILARITY_2 = "data/processed/retrained_2_model_similarity.csv"
SIMILARITY_3 = "data/processed/retrained_3_model_similarity.csv"
SIMILARITY_4 = "data/processed/retrained_4_model_similarity.csv"
