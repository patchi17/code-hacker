import pandas as pd
import csv
import string
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.tokenize import TreebankWordTokenizer
csvfile = open('Test.csv','rt')
csvFileArray = []
tokenData = nltk.word_tokenize(str(csvfile))
stop_words = set(stopwords.words('english'))
#csvFileArray.append(row)
for field, row in (csv.reader(csvfile, delimiter=';')):
    csvFileArray.append(field)
    tokens = word_tokenize(row)
    print(tokens)
    print(row)
    
    for r in tokens:
        if not r in stop_words:
            #appendFile = open(str(idx) + "_file.txt", 'a')
            appendFile = open("testoken.txt", 'a')
            appendFile.write(r+" ")
            #appendFile.write(str(row))
            #appendFile.write(r)
            appendFile.close()
