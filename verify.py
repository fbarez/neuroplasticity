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
  # We should have number of tokens checked = (5 + 1) * number of initial tokens
  tokens_checked = 6 * len(activation_tokens)

  avg_score = sum(new_activation_scores) / tokens_checked
  
  return (new_activation_tokens, avg_score)

