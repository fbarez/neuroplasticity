# !/bin/bash
# Please run: chmod +x src/data/download_data.sh
# Modify script as necessary to extract appropriate concept

wget https://neurox.qcri.org/projects/downloads/bert-concept-net_v1.tgz -O data/external/bert_concept_net

tar xvzf bert-concept-net_v1.tgz

# To extract labels and tokens for location concept
python3 data/external/bert_concept_net/filter_dataset.py SEM:named_entity:location --output_file_prefix data/interim/

# Labels and tokens will be extracted as labels.txt and tokens.txt respectively
# For faster processing, get the first 2500 lines of labels and tokens
head -n2500 labels.txt > location_labels.txt
head -n2500 tokens.txt > location_tokens.txt