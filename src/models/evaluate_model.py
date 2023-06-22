from evaluate import evaluator
from src import tokenizer


def evaluate(chosen_model, dataset):
    task_evaluator = evaluator("token-classification")
    eval_results = task_evaluator.compute(
        model_or_pipeline=chosen_model,
        tokenizer=tokenizer,
        data=dataset,
        metric="seqeval",
    )
    print(eval_results)
