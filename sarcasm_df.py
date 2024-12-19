#load libraries
import pandas as pd
from pathlib import Path
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
from nltk import tokenize
from nltk.tokenize import WordPunctTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import reader
import sentiment
import re
import sarcasm_functions


# Read file
sarcasm_df = reader.process_file_df('/home/m/mchakrav/osso6500/scratch/nlp/train-balanced-sarcasm.csv')

# Tokenize
sarcasm_df['comment'] = sarcasm_df['comment'].astype(str)
sarcasm_df['tokenized_comment'] = sarcasm_df['comment'].apply(sarcasm_functions.custom_tokenizer)

# === Shared Features === 
sarcasm_df['punctuation'] = sarcasm_df['comment'].apply(sarcasm_functions.punc)
sarcasm_df['has_all_caps'] = sarcasm_df['comment'].apply(sarcasm_functions.CAP)
sarcasm_df['positive_emoticon'] = sarcasm_df['comment'].apply(sarcasm_functions.POS_emot)
sarcasm_df['negative_emoticon'] = sarcasm_df['comment'].apply(sarcasm_functions.NEG_emot)
sarcasm_df['hyperbole'] = sarcasm_df['comment'].apply(sarcasm_functions.has_hyperbole)

#add sentiment analysis columns for target and parent comment
sarcasm_df = sentiment.sentiment_df(sarcasm_df)
sarcasm_df = sentiment.sentiment_df(sarcasm_df, parent=True)

sarcasm_df['delta_neg'] = sarcasm_df['neg'] - sarcasm_df['neg_parent']
sarcasm_df['delta_neu'] = sarcasm_df['neu'] - sarcasm_df['neu_parent']
sarcasm_df['delta_pos'] = sarcasm_df['pos'] - sarcasm_df['pos_parent']
sarcasm_df['delta_compound'] = sarcasm_df['compound'] - sarcasm_df['compound_parent']
sarcasm_df['target_pos_parent_neg'] = sarcasm_df['pos'] - sarcasm_df['neg_parent']
sarcasm_df['target_neg_parent_pos'] = sarcasm_df['neg'] - sarcasm_df['pos_parent']
#remove raw text 
shared_features = sarcasm_df[['label','punctuation', 'has_all_caps', 'positive_emoticon', 'negative_emoticon',
       'hyperbole', 'neg', 'neu', 'pos', 'compound', 'neu_parent',
       'pos_parent', 'neg_parent', 'compound_parent','delta_neg','delta_neu','delta_pos','delta_compound','target_pos_parent_neg','target_neg_parent_pos']]

# === TF-IDF Matrix ===
tfidf = TfidfVectorizer(
    tokenizer=sarcasm_functions.identity_tokenizer,
    preprocessor=lambda x: x,
    ngram_range=(1, 3),
    max_features=1000
)

#get tfidf_matrix and put into format 
tfidf_matrix = tfidf.fit_transform(X_train['tokenized_comment'].tolist())
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Combine TF-IDF with shared features
tfidf_feature_matrix = pd.concat([tfidf_df, shared_features], axis=1)

# Save TF-IDF feature matrix
tfidf_feature_matrix.to_csv("feature_matrix_tfidf.tsv", sep='\t', index=False)
#save feature matrix with no n-grams
shared_features.to_csv("features_matrix_notext.tsv",sep='\t',index=False)