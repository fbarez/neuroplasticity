from describe import summarise_neuron

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