from src import basic_model_path
from transformers import AutoModelForTokenClassification
import os


def get_basic_model(model_trainer):
    if os.path.exists(basic_model_path):
        # Get basic model for named entity recognition on pretrained DistilBert
        print("Loading saved basic model...")
        model = AutoModelForTokenClassification.from_pretrained(
            basic_model_path,
            id2label=model_trainer.id2label,
            label2id=model_trainer.label2id,
        )
        return model
    else:
        print("Training basic model...")
        basic_model = model_trainer.train_basic_model()
        basic_model.save_pretrained(basic_model_path)
        return basic_model


def get_retrained_model(pruned_path, model_trainer):
    if os.path.exists(pruned_path):
        # Get basic model for named entity recognition on pretrained DistilBert
        print("Loading saved retrained model...")
        retrained_model = AutoModelForTokenClassification.from_pretrained(
            pruned_path,
            id2label=model_trainer.id2label,
            label2id=model_trainer.label2id,
        )
        return retrained_model
    else:
        print("Retraining pruned model...")
        retrained_model = model_trainer.retrain_pruned_model(pruned_path)
        retrained_model.save_pretrained(pruned_path)
        return retrained_model
