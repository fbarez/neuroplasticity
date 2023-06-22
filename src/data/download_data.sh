# !/bin/bash
# Please run: chmod +x src/data/download_data.sh

wget https://neurox.qcri.org/projects/downloads/bert-concept-net_v1.tgz -O data/external/bert_concept_net

tar xvzf bert-concept-net_v1.tgz

python3 data/external/bert_concept_net/filter_dataset.py SEM:location --output_file_prefix data/interim/location