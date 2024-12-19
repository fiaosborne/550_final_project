#load libraries
import pandas as pd
from pathlib import Path
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import word_tokenize
from nltk import tokenize
from nltk.tokenize import WordPunctTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import reader
import re

#Selin comment: some methods to detect Punctuation, capitalization, emoticons
#I will modify the return depending on how we pair it with the rest of the methods!
#inspo: (Majumdar et al., 2022) for punctuation and upper case list

#Selin: Punctuation
#Overused Cases of Punctuation Marks: !!, !?, ??, ?!, ...

PT = WordPunctTokenizer()

def punc(comment):
  tokens = PT.tokenize(comment)
  count = 0
  for token in tokens:
    if token in ['!!', '!?', '??', '?!', '...']:
      count += 1
  if count >= 1:
    return True
  else:
    return False

#Selin: Presence of capitalization
#Uppercase Words (Excluding Abbreviations And ‘I’ And ‘A’)
def CAP(comment):
  tokens = PT.tokenize(comment)
  count = 0
  for token in tokens:
    if token.isupper():
      if token not in ['I', 'A']:
        count += 1
  if count >= 1:
    return True
  else:
    return False

#Selin: Presence of emoticons
#pos:  :) :D =) ;) :-) ;-) :-D =D ; P =] 8) (:
#neg: :( :/ :‘) :‘( :-( D: ;( :-/ :| :\

def POS_emot(comment):
  tokens = PT.tokenize(comment)
  count = 0
  for token in tokens:
    if token in [':)', ':D', '=)', ';)', ':-)', ';-)', ':-D', '=D', ';P', '=]', '8)', '(:']:
      count += 1
  if count >= 1:
    return True
  else:
    return False

def NEG_emot(comment):
  tokens = PT.tokenize(comment)
  count = 0
  for token in tokens:
    if token in [":(", ":/", ":‘)", ":‘(", ":-(", "D:", ";(", ":-/", ":|"]:
      count += 1
  if count >= 1:
    return True
  else:
    return False

def has_hyperbole(sentence):
  hyperbole_list = [
    # Extreme Magnitudes
    "Absolutely", "Completely", "Entirely", "Exceptionally", "Exorbitantly", 
    "Incredibly", "Insanely", "Outrageously", "Totally", "Unbelievably",
    
    # Extreme Quantities
    "A million", "Billions", "Endless", "Infinite", "Limitless", 
    "Never-ending", "Too many to count", "Tons", "Thousands upon thousands", "Zillions",
    
    # Extreme Comparisons
    "Larger than life", "Bigger than the universe", "Brighter than the sun", 
    "Colder than Antarctica", "Faster than lightning", "Hotter than the sun", 
    "Stronger than steel", "Taller than a skyscraper",
    
    # Time Exaggerations
    "Forever", "Ages", "An eternity", "In the blink of an eye", 
    "A split second", "In no time", "A second feels like a year", "A lifetime",
    
    # Intensity/Emotion
    "Deathly", "Heart-stopping", "Over the moon", "Scared to death", 
    "Shaking like a leaf", "Beyond belief", "I’m dying", "Can’t take it anymore", 
    "I’ve never seen anything like it", "Mind-blowing",
    
    # Other Hyperbolic Expressions
    "The best thing ever", "The worst day of my life", "Out of this world", 
    "Once in a lifetime", "Over the edge", "Blown away", "Knocked me off my feet", 
    "Took my breath away", "Beyond words", "Can’t even describe it"
  ]
  hyperbole_count = 0
  for hyperbole in hyperbole_list:
    if hyperbole in sentence:
      hyperbole_count += 1
  return hyperbole_count

def custom_tokenizer(text):
    # Define a regular expression pattern
    # Matches sequences of punctuation (e.g., ":)"), words (with case sensitivity), or standalone punctuation
    pattern = r"[A-Za-z0-9]+|[:;.,!?)(P|/D-]+"
    # Find all matches using re.findall
    tokens = re.findall(pattern, text)
    return tokens

def identity_tokenizer(text):
    return text  # Assumes text is already a list of tokens

