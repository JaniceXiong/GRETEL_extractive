# GRETEL: Graph Contrastive Topic Enhanced Language Model for Long Document Extractive Summarization

**This code is the implementation of GRETEL

**Python version**: This code is in Python3.6

**Package Requirements**: torch==1.1.0 pytorch_transformers tensorboardX multiprocess pyrouge

Some codes are borrowed from PreSumm (https://github.com/nlpyang/PreSumm)

### Step 1. Download datasets 
#### CORD-19 dataset
Download and unzip the `CORD-19` directories from [here](https://allenai.org/data/cord-19). Put all files in the directory `./raw_data`
#### PubMed dataset
Download zip file from [here] (https://drive.google.com/file/d/1lvsqvsFi3W-pE1SqNZI0s8NR9rC1tsja/view). You can also use the command below to download the files via the cli using linux. Put all files in directory `./raw`.

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lvsqvsFi3W-pE1SqNZI0s8NR9rC1tsja' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lvsqvsFi3W-pE1SqNZI0s8NR9rC1tsja" -O pubmed.zip && rm -rf /tmp/cookies.txt
```
####  S2ORC dataset
Details of the dataset can be found [here] (https://github.com/allenai/s2orc). To prepare, follow instructions [here] (src/datasets/s2orc/README.md))

###  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile (`/.bashrc` file):
```
 for file in `find /home/name/stanford-corenlp-4.2.1  -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-4.2.0` directory. 

###  Step 3. Cleaning data and and Tokenization

For CORD19 or S2ORC data (both from allenai), use the following command to preprocess the data, the raw data files should be in a folder ./raw_data/pmc_data/document_parses/pmc_json/ with the associated metadata csv file at ./raw_data/pmc_data/document_parses/metadata.csv

```
python src/preprocess.py -mode tokenize_allenai_datasets -raw_path ./raw_data/ -save_path ./token_data/ -log ./tokenize_allenai.log
```

For the PubMed dataset. 
```
python src/preprocess.py -mode tokenize_pubmed_dataset -raw_path ./raw/ -save_path ./token_data/ -log ./tokenize_pubmed.log
```

* `RAW_PATH` is the directory containing story files, `save_path` is the target directory to save the generated tokenized files

###  Step 5. Format to Simpler Json Files
 
```
python src/preprocess.py -mode format_to_lines -raw_path ./token_data/ -save_path ./json_data -log ./tokenize.log
```

* `RAW_PATH` is the directory containing tokenized files, `JSON_PATH` is the target directory to save the generated json files

* You can directly use the Json Files provided by us or re-process by yourself according to the above steps

###  Step 6. Generate BoW features from Json Files
 
```
python src/prepro/preprocess_topic_model.py -raw_path ./json_data/ -save_path ./json_data 
```

* `RAW_PATH` is the directory containing json files, `save_path` is the same path as the raw path


###  Step 6. Format to PyTorch Files
```
python src/preprocess.py -mode format_to_bert -raw_path ./json_data/ -save_path ./bert_data/  -lower -n_cpus 1 -log_file ./logs/preprocess.log 
```

* `JSON_PATH` is the directory containing json files, `BERT_DATA_PATH` is the target directory to save the generated binary files
* Note depending on model type you want to use, you can change `format_to_bert` to `format_to_pubmed_bert` or `format_to_robert` or `format_to_biobert`

### Step 8. Model Training
```
python src/train.py -task ext -mode train -bert_data_path ./bert_data/ -ext_dropout 0 -model_path ./models/ -lr 2e-3 -visible_gpus 2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 4000 -train_steps 50000 -accum_count 2 -log_file ./logs/ext_bert_covid -use_interval true -warmup_steps 5000 -model pubmed -topic_number 700 -max_pos 6000
```
### Step 9. Model Evaluation
```
python src/train.py -task ext -mode validate -batch_size 4000 -test_batch_size 4000 -bert_data_path ./bert_data/ -log_file ./logs/val_ext_bert_covid -model_path ./models/ -sep_optim true -use_interval true -visible_gpus 1 -max_pos 6000 -result_path ./results/ext_bert_covid -test_all True -model pubmed -block_trigram False
```
```
python src/train.py -task ext -mode test -batch_size 4000 -test_batch_size 4000 -bert_data_path ./bert_data/ -log_file ./logs/test_ext_bert_covid -test_from ./models/model_step_9000.pt -sep_optim true -use_interval true -visible_gpus 1 -max_pos 6000 -result_path ./results/ext_bert_covid -model pubmed -block_trigram False
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use (choose the top checkpoint on the validation dataset)
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries

