from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from src import tokenizer, MODEL_CHECKPOINT
from src.features.build_features import (
    get_token_class_label_names,
    tokenize_token_class_dataset,
)
import numpy as np
import evaluate


class ModelTrainer:
    def __init__(self):
        self.label_names = get_token_class_label_names()
        self.id2label = {i: label for i, label in enumerate(self.label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.tokenized_dataset = tokenize_token_class_dataset()
        tokenizer.pad_token = tokenizer.eos_token
        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def train_basic_model(self):
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_CHECKPOINT,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        args = TrainingArguments(
            "basic_model",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=5,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=self.__compute_metrics,
            tokenizer=tokenizer,
        )

        trainer.train()
        return model


    def retrain_pruned_model(self, pruned_model_path):
        model = AutoModelForTokenClassification.from_pretrained(
            pruned_model_path,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        args = TrainingArguments(
            "retrained_model",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=12,
            weight_decay=0.01,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=self.__compute_metrics,
            tokenizer=tokenizer,
        )
        trainer.train()
        return model


    def retrain_model_incr(self, pre_model_path):
        model = AutoModelForTokenClassification.from_pretrained(
            pre_model_path,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            
        args = TrainingArguments(
            "retrained_model",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=2,
            weight_decay=0.01,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=self.__compute_metrics,
            tokenizer=tokenizer,
        )
        trainer.train()
        return model


    def __compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        metric = evaluate.load("seqeval")
        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.label_names[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
