from models.train_model import train_model
from features.build_features import load_token_class_dataset
from models.evaluate_model import evaluate
from models.prune_model import prune_model
from src import basic_model_path, pruned_model_path, retrained_model_path


def visualise():
    dataset = load_token_class_dataset()
    # Train and evaluate a basic model on pretrained DistilBert
    model_trainer = train_model()
    basic_model = model_trainer.train_basic_model()
    basic_model.save_pretrained(basic_model_path)
    evaluate(basic_model, dataset["validation"])
    # TODO: Identify neurons in the basic neuron to ablate
    neurons_to_prune = []
    # Ablate neurons
    pruned_model = prune_model(basic_model_path, model_trainer, neurons_to_prune)
    pruned_model.save_pretrained(pruned_model_path)
    evaluate(pruned_model, dataset["validation"])
    # Retrain the pruned model
    retrained_model = model_trainer.retrain_pruned_model(pruned_model_path)
    retrained_model.save_pretrained(retrained_model_path)
    # TODO: Examine the retrained model for concepts
