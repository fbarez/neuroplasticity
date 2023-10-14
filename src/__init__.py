from transformers import AutoTokenizer

# Model specific constants
NEURONS_PER_LAYER = 768
NUM_LAYERS = 6 # DistilBERT, DistilGPT2
# MODEL_CHECKPOINT = "distilbert-base-cased"
MODEL_CHECKPOINT = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

BASIC_MODEL_PATH = "models/basic_model"
PRUNED_MODEL_PATH = "models/pruned_model"
RETRAINED_MODEL_PATH = "models/retrained_model"

CONCEPT_LABEL = "SEM:named_entity:location"
TOKENS_INPUT_PATH = "data/interim/location_tokens.txt"
TOKENS_LABEL_PATH = "data/interim/location_labels.txt"
# CONCEPT_LABEL = "SEM:origin:north_america"
# TOKENS_INPUT_PATH = "data/interim/america_tokens.txt"
# TOKENS_LABEL_PATH = "data/interim/america_labels.txt"
BASIC_ACTIVATIONS_PATH = "data/interim/basic_activations.json"
RETRAINED_ACTIVATIONS_PATH = "data/interim/retrained_activations.json"

BASE_CONCEPT_PATH = "data/processed/basic_model_concepts.csv"
PROCESSED_1 = "data/processed/retrained_model_1_concepts.csv"
PROCESSED_2 = "data/processed/retrained_model_2_concepts.csv"
PROCESSED_3 = "data/processed/retrained_model_3_concepts.csv"
PROCESSED_4 = "data/processed/retrained_model_4_concepts.csv"

SIMILARITY_1 = "data/processed/retrained_1_model_similarity.csv"
SIMILARITY_2 = "data/processed/retrained_2_model_similarity.csv"
SIMILARITY_3 = "data/processed/retrained_3_model_similarity.csv"
SIMILARITY_4 = "data/processed/retrained_4_model_similarity.csv"
