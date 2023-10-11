from src import BASIC_MODEL_PATH
from transformers import AutoModelForTokenClassification, AutoModelForCausalLM
import os


def get_basic_model(model_trainer):
    if os.path.exists(BASIC_MODEL_PATH):
        # Get basic model for named entity recognition on pretrained DistilBert
        print("Loading saved basic model...")
        model = AutoModelForTokenClassification.from_pretrained(
            BASIC_MODEL_PATH,
            id2label=model_trainer.id2label,
            label2id=model_trainer.label2id,
        )
        return model
    else:
        print("Training basic model...")
        basic_model = model_trainer.train_basic_model()
        basic_model.save_pretrained(BASIC_MODEL_PATH)
        return basic_model

def get_pruned_model(pruned_path, model_trainer):
    if os.path.exists(pruned_path):
        # Get basic model for named entity recognition on pretrained DistilBert
        print("Loading saved pruned model...")
        pruned_model = AutoModelForTokenClassification.from_pretrained(
            pruned_path,
            id2label=model_trainer.id2label,
            label2id=model_trainer.label2id,
        )
        return pruned_model
    else:
        print("No saved pruned model. Please prune from basic model!")

def get_retrained_model(retrained_path, pruned_path, model_trainer):
    if os.path.exists(retrained_path):
        # Get basic model for named entity recognition on pretrained DistilBert
        print("Loading saved retrained model...")
        retrained_model = AutoModelForTokenClassification.from_pretrained(
            retrained_path,
            id2label=model_trainer.id2label,
            label2id=model_trainer.label2id,
        )
        return retrained_model
    else:
        print("Retraining pruned model...")
        retrained_model = model_trainer.retrain_pruned_model(pruned_path)
        retrained_model.save_pretrained(retrained_path)
        return retrained_model

def get_incr_retrained_model(post_model_path, pre_model_path, model_trainer):
    if os.path.exists(post_model_path):
        # Get basic model for named entity recognition on pretrained DistilBert
        print("Loading saved retrained model...")
        retrained_model = AutoModelForTokenClassification.from_pretrained(
            post_model_path,
            id2label=model_trainer.id2label,
            label2id=model_trainer.label2id,
        )
        return retrained_model
    else:
        print("Retraining pruned model...")
        retrained_model = model_trainer.retrain_model_incr(pre_model_path)
        retrained_model.save_pretrained(post_model_path)
        return retrained_model
