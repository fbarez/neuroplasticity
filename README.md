# Repository for Automated Identification of Potential Feature Neurons

> Tool to automatically identify potential feature neurons in language models for further investigation.

This repository introduces an automated method of identifying and verifying potential feature neurons in language models, based on analysing maximum activation tokens over a diversified dataset.

## Usage instructions

Configure desired parameters in the Juypter notebook (`auto-feature-neuron-identification.ipynb`):
- Model layer number (middle layers are recommended)
- Model name, e.g. "solu-8l-pile"
- Number of neurons to analyse at once from the model layer
- Similarity threshold (between -1 and 1), i.e. how similar do phrases which activate a neuron need to be, in order for this neuron to be classed as a potential feature neuron?
- Normalised activation threshold (between 0 and 1), i.e. what is the minimum activation score (relative to other neurons analysed) required for a potential feature neuron?
- Number of synonyms to substitute for each word in a maximum activation phrase during verification

Run the code block containing the `id_feature_neuron_results` function, which will return a `pandas` dataframe of results. Each row represents a potential feature neuron and shows its index, average activation score, a description of its maximum activation phrases, and the maximum activation phrases.

## Development setup

Set up a virtual Python environment using `conda`. Import the dependencies listed in `environment.yml`.

```sh
conda env create -n ENVNAME --file environment.yml
```

## Process overview

1. Identification: Detect neurons which activate on similar tokens.

    Scrape the top 20 text samples which activate each neuron the most from Neuroscope, as well as their activation scores.

    Identify the maximum activation token (MAT) in each text sample and its surrounding tokens (5 previous and consecutive tokens) for context. The surrounding tokens constitute the maximum activation phrase (MAP).

    Calculate similarity score among MAPs. Use FastText to compare every MAP to every other MAP in the set of text samples, and average similarity scores.

    Flag neuron for further investigation if similarity between phrases is above a certain threshold.

2. Verification: Verify the type of input which causes neuron activation.

    Retrieve top 5 most similar tokens for each word in each MAP using FastText.

    Substitute each synonym into the original sentence and measure the new activation score for this modified phrase. (If activation score increases, we can include the new phrase as an MAP during the generation phase.)

    Calculate average activation score over total number of phrase variations checked. 

    Normalise activation scores by dividing each score by the maximum activation score found among all neurons in a network layer.

    Neurons with a normalised score above a certain threshold are displayed as potential feature neurons.

3. Generation: Generate description of relationship between activation tokens.

    Connect to OpenAI’s GPT-3 API and prompt it to generate a description for the common relationship between the list of MATs.

## License

Distributed under the MIT license. See ``LICENSE`` for more information.
