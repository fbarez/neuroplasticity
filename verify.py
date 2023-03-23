from identify_neurons import get_max_activations

import gensim.downloader as api
from gensim.models.fasttext import FastText

try:
    model = FastText.load("fasttext.model")
except:
    corpus = api.load("text8")
    model = FastText(corpus)
    model.save("fasttext.model")


def calculate_similarity(activation_tokens):
    """Calculate similarity score among words which activate neuron the most."""
    similarities = []
    for i in range(len(activation_tokens)):
        word = activation_tokens[i]
        try:
            for j in range(i, len(activation_tokens)):
                word_b = activation_tokens[j]
                similarities.append(model.wv.similarity(word, word_b))
        except:
            continue
    if len(similarities) == 0:
        return 0
    return sum(similarities) / len(similarities)


def calculate_phrase_similarity(max_phrases):
    """Calculate similarity score among phrases which activate neuron the most."""
    similarities = []
    for i in range(len(max_phrases)):
        phrase = max_phrases[i]
        try:
            for j in range(i, len(max_phrases)):
                phrase_b = max_phrases[j]
                similarities.append(
                    model.wv.n_similarity(phrase.split(), phrase_b.split())
                )
        except:
            continue
    if len(similarities) == 0:
        return 0
    return sum(similarities) / len(similarities)


def substitute_similar(activation_tokens, cur_acts, index, neuron_layer, NUM_SYNONYMS):
    """Substitute synonyms in for max activation tokens."""
    higher_toks = []
    higher_acts = []
    for i in range(len(activation_tokens)):
        token = activation_tokens[i]
        phrase = activation_tokens[i]
        cur_act = cur_acts[i]
        # Get the top N most similar tokens and check activations
        most_similar = model.wv.most_similar(token, topn=NUM_SYNONYMS)
        similar_tokens = [ls[0] for ls in most_similar]
        for sub in similar_tokens:
            new_phrase = phrase.replace(token, sub)
            new_max_activations = get_max_activations(new_phrase, neuron_layer, index)
            if new_max_activations == None:
                break
            tok, act = new_max_activations
            #  Save tokens which do not decrease activation
            if tok == sub and act >= cur_act:
                higher_toks.append(sub)
                higher_acts.append(act)

    new_activation_tokens = activation_tokens + higher_toks
    new_activation_scores = cur_acts + higher_acts
    tokens_checked = (NUM_SYNONYMS + 1) * len(activation_tokens)

    avg_score = sum(new_activation_scores) / tokens_checked

    return (new_activation_tokens, avg_score)


def substitute_similar_phrases(
    activation_phrases, cur_acts, index, neuron_layer, NUM_SYNONYMS
):
    """Substitute synonyms in for max activation phrases."""
    higher_phrases = []
    higher_acts = []

    for i in range(len(activation_phrases)):
        phrase = activation_phrases[i]
        cur_act = cur_acts[i]
        # For each word in the phrase, get the top N synonyms and check activations
        for word in phrase:
            most_similar = model.wv.most_similar(
                word, topn=NUM_SYNONYMS
            )  # Returns (key, similarity)
            similar_tokens = [ls[0] for ls in most_similar]  # Extract similarity only
            for sub in similar_tokens:
                new_phrase = phrase.replace(word, sub)
                new_max_activations = get_max_activations(
                    new_phrase, neuron_layer, index
                )
                if new_max_activations == None:
                    break
                tok, act = new_max_activations
                #  Save changed phrases which do not decrease activation
                if act >= cur_act:
                    higher_phrases.append(new_phrase)
                    higher_acts.append(act)

    new_activation_phrases = activation_phrases + higher_phrases
    new_activation_scores = cur_acts + higher_acts
    phrases_checked = (NUM_SYNONYMS + 1) * len(activation_phrases)

    avg_score = sum(new_activation_scores) / phrases_checked

    return (new_activation_phrases, avg_score)
