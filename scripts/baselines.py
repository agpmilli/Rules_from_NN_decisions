import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re
import nltk
import numpy as np
from spacy.lang.en import English
from sklearn.model_selection import cross_val_score, KFold, StratifiedShuffleSplit

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "'ve"]


# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")
    
    # replace twitter @mentions
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mentionFinder.sub("@MENTION", text)
    
    # replace twitter #hashtags
    hashtagsFinder = re.compile(r"#[a-z0-9_]{1,30}", re.IGNORECASE)
    text = hashtagsFinder.sub("#HASHTAGS", text)
    
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.IGNORECASE)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ' ')    
    
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    
    # lowercase
    text = text.lower()
    
    return text  
    
# Every step in a pipeline needs to be a "transformer". 
# Define a custom transformer to clean text using spaCy
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# A custom function to tokenize the text using spaCy and convert to lemmas
def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    while "\\n\\n" in tokens:
        tokens.remove("\\n\\n")
    while "\\n" in tokens:
        tokens.remove("\\n")

    return tokens
    
def printNMostInformative(vectorizer, clf, N):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("AntiTrump best: ")
    for feat in topClass1:
        print(feat)
    print("ProTrump best: ")
    for feat in topClass2:
        print(feat)
        
df = pd.read_csv("/mnt/storage01/milliet/data/big/train.csv", index_col=0, encoding = "ISO-8859-1")
train = df[['body','label']]
train.columns = ['text','label']

df2 = pd.read_csv("/mnt/storage01/milliet/data/big/test.csv", index_col=0, encoding = "ISO-8859-1")
test = df2[['body','label']]
test.columns = ['text','label']
        
all = train.append(test, ignore_index=True)
# data

bodyAll = all.text
labelsAll = all.label


parser = English()

# the vectorizer and classifer to use
# note that I changed the tokenizer in CountVectorizer to use a custom function using spaCy's tokenizer
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(2,2), min_df=1000)
#print("LOGISTIC REGRESSION")
#clf = LogisticRegression()
#print("SVM (linear SVC)")
#clf = LinearSVC()
print("SGD")
clf = SGDClassifier()
# the pipeline to clean, tokenize, vectorize, and classify
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])

accuracies = []

bodyTrain = train.text
labelsTrain = train.label

bodyTest = test.text
labelsTest = test.label
    
print("TRAINING...")
# train
pipe.fit(bodyTrain, labelsTrain)

print("PREDICT...")
# test
preds = pipe.predict(bodyTest)

# get the features that the vectorizer learned (its vocabulary)
vocab = vectorizer.get_feature_names()
print(len(vocab))

print("----------------------------------------------------------------------------------------------")
print("results:")
#for (sample, pred) in zip(test, preds):
#    print(sample, ":", pred)
accuracy = accuracy_score(labelsTest, preds)
accuracies.append(accuracy)
print("accuracy:", accuracy)

print("----------------------------------------------------------------------------------------------")
print("Top 10 features used to predict: ")
# show the top features
printNMostInformative(vectorizer, clf, 10)

#print(accuracies)