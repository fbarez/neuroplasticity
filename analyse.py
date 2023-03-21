from describe import summarise_neuron
import pandas as pd

# Iterate through each neuron in a layer of the given model with parameters
def identify_neurons(neuron_layer, max_index, model_name, SIMILARITY_THRESHOLD, NUM_SYNONYMS):
  act_desc = []

  for index in range(max_index):
    summary = summarise_neuron(index, neuron_layer, model_name, SIMILARITY_THRESHOLD, NUM_SYNONYMS)
    if summary is None:
      continue
    avg_act, description, new_tokens = summary
    act_desc.append((avg_act, description, new_tokens, index))

  return act_desc

# Extract results from identifying high potential feature neurons
def analyse_model_layer(layer, model_name, num_neurons, SIMILARITY_THRESHOLD, NORM_ACT_THRESHOLD, NUM_SYNONYMS):
  active_neurons = []

  potential_neurons = identify_neurons(layer, num_neurons, model_name, SIMILARITY_THRESHOLD, NUM_SYNONYMS)

  if len(potential_neurons) == 0:
    return

  max_activation = max([act for act, _, _, _ in potential_neurons])

  for act, desc, new_tokens, index in potential_neurons:
    norm_act = act / max_activation
    if norm_act >= NORM_ACT_THRESHOLD:
      neuron_info = {
          "index": index,
          "avg_activation": norm_act,
          "desc": desc,
          "max_phrases": new_tokens
      }
      active_neurons.append(neuron_info)
  
  return active_neurons

# Save results from analysis in a dataframe
def id_feature_neuron_results(layer, model_name, num_neurons, SIMILARITY_THRESHOLD, NORM_ACT_THRESHOLD, NUM_SYNONYMS):
  active_neurons = analyse_model_layer(layer, model_name, num_neurons, SIMILARITY_THRESHOLD, NORM_ACT_THRESHOLD, NUM_SYNONYMS)
  df = pd.DataFrame(active_neurons)
  return df