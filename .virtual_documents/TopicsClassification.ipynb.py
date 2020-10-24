import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


import re
import os
import sys

import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from bs4 import BeautifulSoup
import unicodedata
from textblob import TextBlob

nlp = spacy.load('en_core_web_sm')


def get_wordcounts(x):
    length = len(str(x).split())
    return length


def get_charcounts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)


def get_avg_wordlength(x):
    count = get_charcounts(x) / get_wordcounts(x)
    return count


def get_stopwords_counts(x):
    l = len([t for t in x.split() if t in stopwords])
    return l


def get_hashtag_counts(x):
    l = len([t for t in x.split() if t.startswith('#')])
    return l


def get_mentions_counts(x):
    l = len([t for t in x.split() if t.startswith('@')])
    return l


def get_digit_counts(x):
    return len([t for t in x.split() if t.isdigit()])


def get_uppercase_counts(x):
    return len([t for t in x.split() if t.isupper()])


def cont_exp(x):
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how does",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        " u ": " you ",
        " ur ": " your ",
        " n ": " and ",
        "won't": "would not",
        'dis': 'this',
        'bak': 'back',
        'brng': 'bring'}

    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x


def get_emails(x):
    emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x)
    counts = len(emails)

    return counts, emails


def remove_emails(x):
    return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', "", x)


def get_urls(x):
    urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=get_ipython().run_line_magic("&:/~+#-]*[\w@?^=%&/~+#-])?',", " x)")
    counts = len(urls)

    return counts, urls


def remove_urls(x):
    return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=get_ipython().run_line_magic("&:/~+#-]*[\w@?^=%&/~+#-])?',", " '', x)")


def remove_rt(x):
    return re.sub(r'\brt\b', '', x).strip()


def remove_special_chars(x):
    x = re.sub(r'[^\w ]+', "", x)
    x = ' '.join(x.split())
    return x


def remove_html_tags(x):
    return BeautifulSoup(x, 'lxml').get_text().strip()


def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


def remove_stopwords(x):
    return ' '.join([t for t in x.split() if t not in stopwords])


def make_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)

    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)


def get_value_counts(df, col):
    text = ' '.join(df[col])
    text = text.split()
    freq = pd.Series(text).value_counts()
    return freq


def remove_common_words(x, freq, n=20):
    fn = freq[:n]
    x = ' '.join([t for t in x.split() if t not in fn])
    return x


def remove_rarewords(x, freq, n=20):
    fn = freq.tail(n)
    x = ' '.join([t for t in x.split() if t not in fn])
    return x


def spelling_correction(x):
    x = TextBlob(x).correct()
    return x



df_all = pd.read_csv('ted_talks_en.csv')


df_all.head()


df = df_all[['transcript', 'topics']].copy()


df.head()


df.loc[0, 'topics']


import  ast


ast.literal_eval(df.loc[0, 'topics'])


df.loc[:, 'topics'] = df.loc[:, 'topics'].apply(lambda x: ast.literal_eval(x))


df.loc[0, 'topics']


df.loc[:,'transcript'] = df.loc[:,'transcript'].apply(lambda x: remove_emails(x))
df.loc[:,'transcript'] = df.loc[:,'transcript'].apply(lambda x: remove_urls(x))
df.loc[:,'transcript'] = df.loc[:,'transcript'].apply(lambda x: remove_special_chars(x))
df.loc[:,'transcript'] = df.loc[:,'transcript'].apply(lambda x: remove_accented_chars(x))
df.loc[:,'transcript'] = df.loc[:,'transcript'].apply(lambda x: remove_stopwords(x))
df.loc[:,'transcript'] = df.loc[:,'transcript'].apply(lambda x: remove_html_tags(x))
df.loc[:,'transcript'] = df.loc[:,'transcript'].apply(lambda x: cont_exp(x.lower()))
# df.loc['transcript'] = df.loc['transcript'].apply(lambda x: spelling_correction(x.lower()))


df.loc[:,'transcript'] = df.loc[:,'transcript'].apply(lambda x: make_base(x))





df.info()





multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df["topics"])


y


y.shape


classes  = multilabel.classes_


bin_topics_df = pd.DataFrame(y, columns=classes)


bin_topics_df


list(bin_topics_df.sum(axis=0).sort_values(ascending=False).head(30).index)


popular_topics = ['science',
 'technology',
 'culture',
 'global issues',
 'society',
 'social change',
 'business',
 'health',
 'history',
 'education',
 'humanity',
 'innovation',
 'biology',
 'entertainment',
 'future',
 'art',
 'communication',
 'creativity',
 'medicine',
 'personal growth',
 'environment',
 'economics']


popular_topics_df = bin_topics_df.loc[:,popular_topics]


popular_topics_df


df = pd.concat([df,popular_topics_df], axis=1)


df.loc[:,'science'].sum()


df = df.loc[~(df[list(popular_topics_df)] == 0).all(axis=1)]


tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=4000)


X= tfidf.fit_transform(df['transcript'])


y = df.loc[:, list(popular_topics_df)]


y


X.shape, y.shape


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 0)


X_train.shape, X_test.shape


#### Since this is a multilabel classifier we need to import the appropriate libraries


from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score


lr = LogisticRegression()


clf = OneVsRestClassifier(lr)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


#[None, 'micro', 'macro', 'weighted', 'samples']
jaccard_score(y_test, y_pred, average='samples')


test_transcript = tfidf.transform(['''the basic technical idea behind deep learning in your 
                                    networks have been around for decades why are they only just now taking off in this video let's go 
                                    over some of the main drivers behind the rise of deep learning because I think this will help you 
                                    that the spot the best opportunities within your own organization'''])
filters = list(clf.predict(test_transcript)[0] == 1)
pred_topics = [i for (i, v) in zip(popular_topics, filters) if v] 
pred_topics


from sklearn.svm import LinearSVC


svm = LinearSVC()
clf = OneVsRestClassifier(svm)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


#[None, 'micro', 'macro', 'weighted', 'samples']
jaccard_score(y_test, y_pred, average='samples')


test_transcript = tfidf.transform(['''the basic technical idea behind deep learning in your 
                                    networks have been around for decades why are they only just now taking off in this video let's go 
                                    over some of the main drivers behind the rise of deep learning because I think this will help you 
                                    that the spot the best opportunities within your own organization'''])
filters = list(clf.predict(test_transcript)[0] == 1)
pred_topics = [i for (i, v) in zip(popular_topics, filters) if v] 
pred_topics
