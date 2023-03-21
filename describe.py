from identify import scrape_neuron_max_activations
from verify import calculate_similarity, calculate_phrase_similarity, substitute_similar, substitute_similar_phrases
import openai

API_KEY="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"
openai.api_key = API_KEY

# Returns a GPT-3 generated summary of potential content.
def ask_summary(keywords_list):
  prompt_summary = "What do all these phrases have in common? Phrases: " + ' '.join(keywords_list) + ". Common:"
  # print(prompt_summary)
  response = openai.Completion.create(engine="text-davinci-003", prompt=prompt_summary, max_tokens=50)
  summary = response["choices"][0]["text"]
  return summary.strip()

def summarise_neuron(index, neuron_layer, model_name):
  tokens, acts, max_phrases = scrape_neuron_max_activations(model_name, neuron_layer, index)
  cur_similarity = calculate_similarity(tokens)
  cur_phrase_similarity = calculate_phrase_similarity(max_phrases)

  if cur_similarity < 0.6 and cur_phrase_similarity < 0.6:
    return
  
  new_avg_act = 0
  try:
    # Test and get the new average activation score: should be higher due to testing more tokens, averaged over fixed number of checked neurons
    # new_tokens, new_avg_tok_act = substitute_similar(
    #     tokens,
    #     acts, 
    #     index,
    #     neuron_layer)
    new_phrases, new_avg_act = substitute_similar_phrases(
        max_phrases,
        acts,
        index,
        neuron_layer
    )
  finally:
    tokens_checked = 6 * len(tokens)
    cur_avg_act = sum(acts) / tokens_checked
    # if new_avg_act >= cur_avg_act:
    #   description = new_phrases
    # else:
    #   description = max_phrases
    # return cur_avg_act, description, tokens

    if new_avg_act >= cur_avg_act:
      description = ask_summary(new_phrases)
      return new_avg_act, description, new_phrases
    else:
      description = ask_summary(tokens)
      description = ask_summary(max_phrases)
      return cur_avg_act, description, tokens
