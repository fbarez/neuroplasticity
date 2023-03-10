import identify, verify

import openai

API_KEY="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" # Open license from alignment jam
openai.api_key = API_KEY

# Returns a GPT-3 generated summary of potential content.
def ask_summary(keywords_list):
  prompt_summary = "What do all these words have in common? Words: " + ' '.join(keywords_list) + ". Common:"
  # print(prompt_summary)
  response = openai.Completion.create(engine="text-davinci-003", prompt=prompt_summary, max_tokens=50)
  summary = response["choices"][0]["text"]
  return summary.strip()

def summarise_neuron(index, neuron_layer, model_name):
  tokens, acts, _ = scrape_neuron_max_activations(model_name, neuron_layer, index)
  cur_similarity = calculate_similarity(tokens)

  if cur_similarity < 0.6:
    return
  
  new_avg_act = 0
  try:
    # Test and get the new average activation score: should be higher due to testing more tokens, averaged over fixed number of checked neurons
    new_tokens, new_avg_act = substitute_similar(
        tokens,
        acts, 
        index,
        neuron_layer)
  finally:
    tokens_checked = 6 * len(tokens)
    cur_avg_act = sum(acts) / tokens_checked

    if new_avg_act >= cur_avg_act:
      description = ask_summary(new_tokens)
      return new_avg_act, description, new_tokens
    else:
      description = ask_summary(tokens)
      return cur_avg_act, description, tokens
    
# Iteration
def identify_neurons(neuron_layer, max_index, model_name):
  act_desc = []

  for index in range(max_index):
    summary = summarise_neuron(index, neuron_layer, model_name)
    if summary is None:
      continue
    avg_act, description, new_tokens = summary
    act_desc.append((avg_act, description, new_tokens, index))

  return act_desc

def analyse_model_layer(layer, model_name, num_neurons):
  act_desc = identify_neurons(layer, num_neurons, model_name)

  if len(act_desc) == 0:
    return

  sorted_by_act = sorted(act_desc, key=lambda tup: tup[1], reverse=True)
  max_activation = max([act for act, _, _, _ in sorted_by_act])
  print("Max activation: ", max_activation)

  for act, desc, new_tokens, index in sorted_by_act:
    norm_act = act / max_activation
    if norm_act >= 0.3:
      print("Neuron " + str(index) + ": activation " + str(norm_act) + ", " + desc)
      print("Tokens: ", new_tokens)