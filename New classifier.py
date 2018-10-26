
# coding: utf-8
#Drug&Gun Nudism and Other Classifier

import os
import re
import csv
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC,NuSVC,LinearSVC  
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import linear_model
from sklearn import manifold
from sklearn.manifold import SpectralEmbedding
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import average_precision_score
import sqlite3 as lt
import seaborn as sns
import string
get_ipython().magic(u'matplotlib inline')
from sklearn.model_selection import GridSearchCV

#Text Preprocessing
def text_process(doc):
    punctuations = '''!()-[]{};:'"\,0123456789<>+./?@#$%^&*_~'''
    no_punct = ""
    for senten in doc:
        if senten not in punctuations:
            no_punct = ((no_punct + senten).lower())
    return  (no_punct)
#Removing Whitespaces
def remove_manyspaces(doc):
    return re.sub(r'\s+', ' ', doc)
#Cleaning Text
def clean_text(doc):
    text = text_process(doc)
    text = remove_manyspaces(doc)
    return doc.lower()
#Stopwords
def rmv_stopwords(doc):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(doc)
    not_stop = " "
    for w in word_tokens: 
        if not w in stop_words:
            not_stop = ((not_stop + " " + w).lower())
    return(not_stop)

#Reading main Dataframe
df1= pd.read_csv('recout.csv', sep=';', encoding='utf8')
df1
#Dropping unwanted rows
df1.drop(['Sites','URL','Title','MCat','CCat'], axis=1,inplace=True)

df1

#Text processing
df1_cleaned = df1['Description'].apply(text_process)
df1['descleaned']= df1_cleaned
df1

#Stopwords
df1_cleaned = df1['descleaned'].apply(rmv_stopwords)
df1['desstop']= df1_cleaned
df1

df1.dtypes

#Splitting values of Scat to new coloumn SCat1
df1['SCat1'] = [x.split()[0] for x in df1['SCat']]

df1

df1.drop(['Description','SCat','descleaned'], axis=1, inplace=True)

df1

df1[(df1.SCat1 == 'Drugs') | (df1.SCat1 == 'Guns')]

df1.loc[(df1.SCat1 == 'Drugs') | (df1.SCat1 == 'Guns'), 'SCat1'] = 'DrugandGun'

df_dandg = df1[df1.SCat1=='DrugandGun']

df_dandg

df1[(df1.SCat1 == 'DrugandGun') | (df1.SCat1 == 'Nudism')]

#Renaming Other
df1.loc[(df1.SCat1 != 'Nudism') & (df1.SCat1 != 'DrugandGun'), 'SCat1'] = 'Other'

#Other
df_other = df1[df1.SCat1=='Other']

df_other

df_other.to_csv('Other.csv')

#Reading a new CSV
df_other = pd.read_csv('Other.csv', sep=';',encoding='utf8')

df_other
#Drugs and Guns
df_dandg = df1[(df1.SCat1 == 'DrugandGun')]
df_dandg

df_dandg.to_csv('Drug&Gun.csv')
#Reading a new CSV
df_dandg = pd.read_csv('Drug&Gun.csv', sep=';',encoding='utf8')

df_dandg
#Nudism
df_Nudism = df1[(df1.SCat1 == 'Nudism')]

df_Nudism

df_Nudism.to_csv('Nudism.csv')

#Reading nudism csv
df_Nudism = pd.read_csv('Nudism.csv', sep=';',encoding='utf8')

df_Nudism

df_other['SCat1'].describe()

df_Nudism['SCat1'].describe()

df_dandg['SCat1'].describe()

df_test = [df_other,df_dandg,df_Nudism]
#Concatination different dataframes
df_full = pd.concat(df_test)
df_full

df_full['SCat1'].describe()
#Mapping labels
df_full['label_SCat'] = df_full.SCat1.map({'Other': 0, 'DrugandGun': 1, 'Nudism': 2 })


#Creating Variables for original training data
X= df_full['desstop']
y= df_full['label_SCat']

X1_train, X1_test, y1_train, y1_test = train_test_split(X,y,test_size=0.5, random_state=1)

pipeline = Pipeline([('vect',TfidfVectorizer(max_features=10000,ngram_range=(4,4), analyzer='word',
                                             tokenizer=text_process,use_idf= 'True')),
                     ('tfidf',TfidfTransformer()),
                     ('clf',MultinomialNB())])

pipeline.fit(X1_train, y1_train)

print (pipeline.score(X1_test, y1_test))

pred = pipeline.predict(X1_test)

print (metrics.classification_report(y1_test, pred))

label_SCat = ['Other','DrugandGuns', 'Nudism']

cm = metrics.confusion_matrix(y1_test, pred)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=label_SCat, yticklabels=label_SCat);
plt.title('Confusion Matrix of the labels')





#Reading Dataset
df3= pd.read_csv('testlabel.csv', sep=';', encoding='utf8')

df3

# text processing

df3_cleaned = df3['desstop'].apply(text_process)

df3['desstop'] = df3_cleaned

df3
#Stopwords
df3_cleaned = df3['desstop'].apply(rmv_stopwords) 

df3['desstop'] = df3_cleaned

df3

#Creating Variables

X_samp= df3['desstop']
y_samp= df3['label_SCat']

#Splitting them into training and test
X_samp_train, X_samp_test, y_samp_train, y_samp_test = train_test_split(X_samp,y_samp,test_size=1, random_state=1)

pipeline = Pipeline([('vect',TfidfVectorizer(max_features=10000,ngram_range=(4,4), analyzer='word', 
                                             tokenizer=text_process,use_idf= 'True')),
                      ('tfidf',TfidfTransformer()),
                      ('clf',MultinomialNB())])

pipeline.fit(X1_train, y1_train)

#Calculating Score
print (pipeline.score(X_samp_test, y_samp_test))

pred2 = pipeline.predict(X_samp_test)
#Classification Report
print (metrics.classification_report(y_samp_test, pred2))

label_SCat = ['Other','DrugandGuns', 'Nudism']

cm2 = metrics.confusion_matrix(y_samp_test, pred2)
#Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(cm2, annot=True, cmap="Blues", fmt="d", xticklabels=label_SCat, yticklabels=label_SCat);
plt.title('Confusion Matrix of the labels')


