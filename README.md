# Setup
Follow these steps to get inference up and running.

## Installing
They used conda but it broke for me so I used venv
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Models
Download the model from her:
* [Sentiment analysis model trained on Financial PhraseBank](https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/finbert-sentiment/pytorch_model.bin)

The workflow should be like this:
* Create a directory for the model. For example: `models/sentiment/pytorch_model.bin`
* Download the model and put it into the directory you just created.
* Put a copy of `config.json` in this same directory. 
* Call the model with `.from_pretrained(<model directory name>)`

## Getting predictions
We provide a script to quickly get sentiment predictions using FinBERT. Given a .txt file, `predict.py` produces a .csv file including the sentences in the text, corresponding softmax probabilities for three labels, actual prediction and sentiment score (which is calculated with: probability of positive - probability of negative).

Here's an example with the provided example text: `test.txt`. From the command line, simply run:
```bash
python predict.py --text_path test.txt --output_dir output/ --model_path models/sentiment/pytorch_model.bin
```

## Other things you need to do
* Install nltk
* Download the punkt tokenizer
```bash
python -m nltk.downloader punkt_tab
```
* Add 'model_type' to the config.json file after you moved the model to the models/sentiment directory. The file should look like this:
```json
{
  "model_type": "bert",
  ...
}
```
* OPTIONAL BUT GOOD PRACTICE: Add 'num_labels' to the config.json file after you moved the model to the models/sentiment directory. The file should look like this:
```json
{
  "num_labels": 3,
  ...
}
```
* GPU support: I added support for Mac integrated GPU so if you run predict.py with --use_gpu flag, it will use the integrated GPU.
```bash
python predict.py --text_path test.txt --output_dir output/ --model_path models/sentiment --use_gpu
```

## Datasets
For the sentiment analysis, we used Financial PhraseBank from [Malo et al. (2014)](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts).
 The dataset can be downloaded from this [link](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list).
 If you want to train the model on the same dataset, after downloading it, you should create three files under the 
 `data/sentiment_data` folder as `train.csv`, `validation.csv`, `test.csv`. 
To create these files, do the following steps:
- Download the Financial PhraseBank from the above link.
- Get the path of `Sentences_50Agree.txt` file in the `FinancialPhraseBank-v1.0` zip.
- Run the [datasets script](scripts/datasets.py):
```python scripts/datasets.py --data_path <path to Sentences_50Agree.txt>```

## Training the model
Training is done in `finbert_training.ipynb` notebook. The trained model will
 be saved to `models/classifier_model/finbert-sentiment`. You can find the training parameters in the notebook as follows:
```python
config = Config(   data_dir=cl_data_path,
                   bert_model=bertmodel,
                   num_train_epochs=4.0,
                   model_dir=cl_path,
                   max_seq_length = 64,
                   train_batch_size = 32,
                   learning_rate = 2e-5,
                   output_mode='classification',
                   warm_up_proportion=0.2,
                   local_rank=-1,
                   discriminate=True,
                   gradual_unfreeze=True )
```
The last two parameters `discriminate` and `gradual_unfreeze` determine whether to apply the corresponding technique 
against catastrophic forgetting.

## Getting predictions
We provide a script to quickly get sentiment predictions using FinBERT. Given a .txt file, `predict.py` produces a .csv file including the sentences in the text, corresponding softmax probabilities for three labels, actual prediction and sentiment score (which is calculated with: probability of positive - probability of negative).

Here's an example with the provided example text: `test.txt`. From the command line, simply run:
```bash
python predict.py --text_path test.txt --output_dir output/ --model_path models/classifier_model/finbert-sentiment
```

# Profiling
To profile the model, run the `finbert_profiling.ipynb` notebook. This is a modified version of the original notebook that adds profiling to the model.

## Model Variants
I've set it up so that we can load different model variants (i.e. with different optimizations) and profile them. The current variants are:
* Baseline
* Quantized Int8

Instructions on how to select between variants to train/run and how to add a new variant are in the notebook.