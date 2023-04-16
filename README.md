# Repository for Investigation of Conceptual Neuroplasticity in Language Models

> Aim: investigate the relationship between groups of concept neurons, using the inherent plasticity of artificial neural networks.

Usage instructions
------------

Functions for training, pruning and evaluating models are in `src/models`. Functions for analysing models using the [NeuroX](https://neurox.qcri.org/docs/index.html#) library are in `src/visualization`.

Variables storing path information, e.g. string paths to saved models, are in `src/__init__.py`. Modify or add new variables related to paths or model-specific metadata here.

See the Juypter notebook at `notebooks/locations.ipynb` for an example of pruning concept neurons related to location names, and examining how this concept reappears after retraining the pruned model.

Development setup
------------

Set up a Python virtual environment and install the dependencies listed in `requirements.txt`.

```sh
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Process overview
------------

We identify a cluster of neurons related to a specific concept, ablate them, and train the model post-ablation.

* Expectation: we can continue training until the model re-learns the same task.
* Expectation: existing neurons learn new functionality and become polysemantic.
* Unknown: which neurons are now responsible for handling subconcepts?
* Unknown: does the same concept still exist, or is it handled by a different process?
* Unknown: if a neuron becomes polysemantic, are the multiple concepts existing in that one neuron similar in any way?
* Unknown: is there a pattern which can be linked to the theory of superposition?

Our main contributions are as follows:

* We demonstrate the synaptic and functional plasticity of artificial neural networks after explicitly pruning salient neurons and fine-tuning.
* We investigate the relationship between groups of concept neurons in terms of how they might learn to take on each others' semantics.
* We present preliminary studies into how we can force specific (?) neurons to become polysemantic by leveraging the plasticity of neural networks during pruning.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## License

Distributed under the MIT license. See ``LICENSE`` for more information.
