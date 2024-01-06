# Repository for "Large Language Models Relearn Removed Concepts"

> Aim: investigate the relationship between groups of concept neurons, using the inherent plasticity of artificial neural networks.

## Usage instructions

Download the data (labels and tokens for a specific concept) from BERT Concept Net. Use `src/data/download_data.sh` to download labels and tokens for the concept of locations - feel free to modify the script if needed.

```sh
./src/data/download_data.sh
```

Run the methodology to generate data about a model which undergoes ablation and retraining. This will build a base model, prune the top concept neurons, retrain the model and generate data about top activating neurons and top activating tokens throughout the process.

```sh
python3 run.py
```

See the Jupyter notebook at `notebooks/analysis.ipynb` to produce graphs and tables to analyze the redistribution of concepts during the process of neuroplasticity.

## Development setup

Set up a Python virtual environment and install the dependencies listed in `requirements.txt`.

```sh
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Functions for training, pruning and evaluating models are in `src/models`. Functions for analysing models using the [NeuroX](https://neurox.qcri.org/docs/index.html#) library are in `src/visualization`.

Variables storing path information, e.g. string paths to saved models, are in `src/__init__.py`. Modify or add new variables related to paths or model-specific metadata here.

See the Jupyter notebook at `notebooks/locations.ipynb` for an example of pruning concept neurons related to location names, and examining how this concept reappears after retraining the pruned model.

### Paper

TODO: add link to paper when released!

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Data for tokens annotated with concept labels.
    │   ├── processed      <- The final, canonical data sets for analysis.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │   └── analysis       <- Produce graphs and tables for analysis of concept redistribution.
    │   └── locations      <- Example of pruning concept neurons for location names and retraining
    │   └── saliency       <- Create concept saliency heatmaps across the model.
    │   └── similarity     <- Create concept similarity heatmaps across the model.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── download_data.sh
    │   │
    │   ├── features       <- Scripts to turn data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── evaluate_model.py
    │   │   └── prune_model.py
    │   │   └── ModelTrainer.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── get_models.py
    │       └── results_utils.py
    │       └── ModelAnalyzer.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## License

Distributed under the MIT license. See `LICENSE` for more information.

Please cite using:
\cite@misc{lo2024large,
      title={Large Language Models Relearn Removed Concepts}, 
      author={Michelle Lo and Shay B. Cohen and Fazl Barez},
      year={2024},
      eprint={2401.01814},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}



