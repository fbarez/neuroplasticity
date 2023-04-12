from transformers import AutoTokenizer

model_max_neurons = 768
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
basic_model_path = "../../models/basic_model"
pruned_model_path = "../../models/pruned_model"
retrained_model_path = "../../models/retrained_model"
