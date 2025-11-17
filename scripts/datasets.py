import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

if not os.path.exists('data/sentiment_data'):
    os.makedirs('data/sentiment_data')

parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--data_path', type=str, help='Path to the text file.')

args = parser.parse_args()
# Try reading with different encodings to handle non-UTF-8 files
# Using engine='python' to avoid ParserWarning for regex separators
try:
    data = pd.read_csv(args.data_path, sep='.@', names=['text','label'], encoding='utf-8', engine='python')
except UnicodeDecodeError:
    try:
        data = pd.read_csv(args.data_path, sep='.@', names=['text','label'], encoding='latin-1', engine='python')
    except UnicodeDecodeError:
        data = pd.read_csv(args.data_path, sep='.@', names=['text','label'], encoding='utf-8', errors='replace', engine='python')

train, test = train_test_split(data, test_size=0.2, random_state=0)
train, valid = train_test_split(train, test_size=0.1, random_state=0)

train.to_csv('data/sentiment_data/train.csv',sep='\t')
test.to_csv('data/sentiment_data/test.csv',sep='\t')
valid.to_csv('data/sentiment_data/validation.csv',sep='\t')