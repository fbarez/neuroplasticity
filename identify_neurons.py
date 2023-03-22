from bs4 import BeautifulSoup

from transformer_lens.utils import to_numpy
from transformer_lens import HookedTransformer

import urllib.request
import re

model_name = "solu-8l-pile"
solu_model = HookedTransformer.from_pretrained(model_name)


def get_neuron_url(model_name, layer, number):
    """Return the URL for Neuroscope's model neuron index."""
    model_index = "https://neuroscope.io/"
    return model_index + model_name + "/" + str(layer) + "/" + str(number) + ".html"


def scrape_neuroscope(neuron_url):
    """Scrape the Neuroscope website for the chosen neuron."""
    neuron_request = urllib.request.urlopen(neuron_url, timeout=120).read()
    return BeautifulSoup(neuron_request)


def clean_scraped_data(dataset):
    """Extract the list of tokens, list of activations and the max activation."""
    data = dataset.text
    tokens_match = re.search(r"\{\"tokens\": \[(.*?)\], \"values", data)
    acts_match = re.search(r"\"values\": \[(.*?)\]", data)

    if tokens_match and acts_match:
        # Convert into clean list of strings
        token_list = tokens_match.group(1).replace('"', "").split(", ")[:-1]
        act_list = acts_match.group(1).replace('"', "").split(",")[:-1]
        # Identify maximum activation score (normalised)
        act_floats = [float(x) for x in act_list]
        maximum = max(act_floats)
        return token_list, act_list, maximum


def scrape_neuron_max_activations(model_name, layer, number):
    """
    Return a list of max activating tokens and their activations for a neuron.

    model_name: name of language model
    layer: layer of model that the neuron is in
    number: index of the neuron in the layer
    """
    neuron_url = get_neuron_url(model_name, layer, number)
    scraped = scrape_neuroscope(neuron_url)

    max_tokens = []
    max_phrases = []
    max_acts = []

    count = 0
    for dataset in scraped.find_all("script", type="module"):
        count += 1
        # Ignore full text scraped. We only want the max activating sentences.
        if count % 2 == 0:
            continue
        # Extract tokens and activations.
        token_list, act_list, maximum = clean_scraped_data(dataset)
        if token_list is None or act_list is None:
            print("Could not retrieve tokens and activations.")
            break
        # Return the max act token, its activation and its surrounding phrase
        index = 0
        for tok, act in zip(token_list, act_list):
            if float(act) == maximum:
                before = max(index - 5, 0)
                after = min(index + 5, len(token_list) - 1)
                max_tokens.append(tok)
                max_acts.append(maximum)
                max_phrases.append(" ".join(token_list[before:after]))
            index += 1

    return (max_tokens, max_acts, max_phrases)


def get_neuron_acts(text, layer, neuron_index):
    """Extract state from a single hook (see Neuroscope documentation)."""
    cache = {}

    def caching_hook(act, hook):
        cache["activation"] = act[0, :, neuron_index]

    solu_model.run_with_hooks(
        text, fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", caching_hook)]
    )
    return to_numpy(cache["activation"])


def get_max_activations(text, layer, neuron_index):
    """
    text: The text to visualize
    layer: The layer index
    neuron_index: The neuron index

    Returns the token with the highest activation in the text, and its activation
    """
    if layer is None:
        return "Please select a Layer"
    if neuron_index is None:
        return "Please select a Neuron"

    acts = get_neuron_acts(text, layer, neuron_index)
    act_max = acts.max()

    # Convert the text to a list of tokens
    str_tokens = solu_model.to_str_tokens(text)

    # Print the max act token and its surrounding phrase
    for tok, act in zip(str_tokens, acts):
        if act == act_max:
            return (tok, act)
