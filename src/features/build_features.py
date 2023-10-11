from datasets import load_dataset
from src import tokenizer


def load_token_class_dataset():
    # return load_dataset("conll2003")
    eli5 = load_dataset("eli5")
    eli5 = eli5.train_test_split(test_size=0.2)
    return eli5.flatten()


def get_token_class_label_names():
    raw_dataset = load_token_class_dataset()
    ner_feature = raw_dataset["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    return label_names


def tokenize_token_class_dataset():
    return __tokenize_token_class_dataset_gpt2()


def __tokenize_token_class_dataset_gpt2():
    tokenized_eli5 = eli5.map(
        preprocess_eli5_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5["train"].column_names,
    )
    return tokenized_eli5.map(group_texts, batched=True, num_proc=4)


def preprocess_eli5_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])


def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def __tokenize_token_class_dataset_bert():
    raw_dataset = load_token_class_dataset()
    tokenized_dataset = raw_dataset.map(
        __tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )
    return tokenized_dataset


def __tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(__align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def __align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
