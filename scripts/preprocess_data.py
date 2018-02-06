import pandas as pd
import string
import re
import nltk
import numpy as np
from spacy.lang.en import English
from spacy.symbols import ORTH, LEMMA, POS
import sys

"""
preprocess-twitter.py
python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"
Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu
Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " <hashtag> {} <allcaps> ".format(hashtag_body.lower())
    else:
        result = " <hashtag> {}".format(hashtag_body.lower())
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> "


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", " <user> ")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ")
    text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ")
    text = re_sub(r"<3"," <heart> ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat> ")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ")

    text = re_sub(r"([A-Z]){2,}", allcaps)

    return ' '.join(text.lower().split())

name = "validate"

df = pd.read_csv("/mnt/storage01/milliet/data/big/" + name + ".csv", index_col=0, encoding = "ISO-8859-1")
df1 = df[['body','label']]
df1.columns = ['text','label']

text_list = df1['text'].tolist()
label_list = df1['label'].tolist()
    
parser = English()
parser.tokenizer.add_special_case(u'<smile>', [{ORTH: u'<smile>'}])
parser.tokenizer.add_special_case(u'<lolface>', [{ORTH: u'<lolface>'}])
parser.tokenizer.add_special_case(u'<sadface>', [{ORTH: u'<sadface>'}])
parser.tokenizer.add_special_case(u'<neutralface>', [{ORTH: u'<neutralface>'}])
parser.tokenizer.add_special_case(u'<heart>', [{ORTH: u'<heart>'}])
parser.tokenizer.add_special_case(u'<url>', [{ORTH: u'<url>'}])
parser.tokenizer.add_special_case(u'<user>', [{ORTH: u'<user>'}])
parser.tokenizer.add_special_case(u'<number>', [{ORTH: u'<number>'}])
parser.tokenizer.add_special_case(u'<allcaps>', [{ORTH: u'<allcaps>'}])
parser.tokenizer.add_special_case(u'<hashtag>', [{ORTH: u'<hashtag>'}])
parser.tokenizer.add_special_case(u'<repeat>', [{ORTH: u'<repeat>'}])
parser.tokenizer.add_special_case(u'<elong>', [{ORTH: u'<elong>'}])

print("Apply cleaning")
new_text_list = []
i=0
size = len(text_list)
for text in text_list:
    print(str(i) + " / " + str(size) + " elements treated.", end='\r')
    tok_text = tokenize(text)
    spacy_text = " ".join([sent.string.strip() for sent in parser(tok_text)])
    while spacy_text.startswith('"'):

        spacy_text = spacy_text[1:]
    while spacy_text.endswith('"'):
        spacy_text = spacy_text[:-1]
    new_text_list.append(spacy_text)
    i+=1
    
print("DONE")

final_df = pd.DataFrame(
    {'text': new_text_list,
     'label': label_list
    })

final_df.to_csv("/mnt/storage01/milliet/data/big/" + name + "_processed.tsv", encoding='utf-8', sep='\t', header=False, index=False)