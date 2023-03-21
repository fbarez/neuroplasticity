from identify import get_max_activations

import gensim.downloader as api
from gensim.models.fasttext import FastText

try:
  model = FastText.load("fasttext.model")
except:
  corpus = api.load('text8')
  model = FastText(corpus)
  model.save("fasttext.model")

# Calculate similarity score among words which activate neuron the most
def calculate_similarity(activation_tokens):
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

# Calculate similarity score among phrases which activate neuron the most
def calculate_phrase_similarity(max_phrases):
  similarities = []
  for i in range(len(max_phrases)):
    phrase = max_phrases[i]
    try:
      for j in range(i, len(max_phrases)):
        phrase_b = max_phrases[j]
        similarities.append(model.wv.n_similarity(phrase.split(), phrase_b.split()))
    except:
      continue
  # Compute cosine similarity between two sentence vectors
  # max_phrases_vectors = [model.wv.get_sentence_vector(x) for x in max_phrases]
  # for i in range(len(max_phrases_vectors)):
  #   phrase_vec = max_phrases_vectors[i]
  #   other_vecs = [x for x in max_phrases_vectors if x != phrase_vec]
  #   try:
  #     sim = model.wv.cosine_similarities(phrase_vec, other_vecs)
  #     similarities.append(sim)
  #   except:
  #     continue
  if len(similarities) == 0:
    return 0
  return sum(similarities) / len(similarities)

def substitute_similar(activation_tokens, cur_acts, index, neuron_layer):
  higher_toks = []
  higher_acts = []
  for i in range(len(activation_tokens)):
    token = activation_tokens[i]
    phrase = activation_tokens[i]
    cur_act = cur_acts[i]
    # Get the top 5 most similar tokens and check activations
    most_similar = model.wv.most_similar(token, topn=5)
    similar_tokens = [ls[0] for ls in most_similar]
    for sub in similar_tokens:
      new_phrase = phrase.replace(token, sub)
      new_max_activations = get_max_activations(
          new_phrase, 
          neuron_layer, 
          index)
      if new_max_activations == None:
        break
      tok, act = new_max_activations
      # Save tokens which do not decrease activation
      if tok == sub and act >= cur_act:
        higher_toks.append(sub)
        higher_acts.append(act)

  new_activation_tokens = activation_tokens + higher_toks
  new_activation_scores = cur_acts + higher_acts
  # We should have number of tokens checked = (9 + 1) * number of initial tokens
  tokens_checked = 10 * len(activation_tokens)

  avg_score = sum(new_activation_scores) / tokens_checked
  
  return (new_activation_tokens, avg_score)

def substitute_similar_phrases(activation_phrases, cur_acts, index, neuron_layer):
  higher_phrases = []
  higher_acts = []

  for i in range(len(activation_phrases)):
    phrase = activation_phrases[i]
    cur_act = cur_acts[i]
    # For each word in the phrase, get the top 5 synonyms and check activations
    for word in phrase:
      most_similar = model.wv.most_similar(word, topn=5) # Returns (key, similarity)
      similar_tokens = [ls[0] for ls in most_similar] # Extract similarity only
      for sub in similar_tokens:
        new_phrase = phrase.replace(word, sub)
        new_max_activations = get_max_activations(
            new_phrase, 
            neuron_layer, 
            index)
        if new_max_activations == None:
          break
        tok, act = new_max_activations
        # Save changed phrases which do not decrease activation
        if act >= cur_act:
          higher_phrases.append(new_phrase)
          higher_acts.append(act)

  new_activation_phrases = activation_phrases + higher_phrases
  new_activation_scores = cur_acts + higher_acts
  # We should have number of phrases checked = (5 + 1) * number of initial tokens
  phrases_checked = 6 * len(activation_phrases)

  avg_score = sum(new_activation_scores) / phrases_checked
  
  return (new_activation_phrases, avg_score)