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


#tokens1 = word_tokenize(row)

#file = open(str(idx) + "_file.txt", 'w+')  # enumerate do same as you want!
#file.write(str(row))
#file.write(str(tokens))
#csvfile.close()
#csvfile1 = open('cleaned.csv','rt')
#for idx, row in enumerate(csv.reader(csvfile, delimiter=';')):
#csvFileArray.append(row)
#print(row)
#file = open(str(idx) + "_file.csv", 'w+')  # enumerate do same as you want!
#file.write(str(tokens))

#reader = csv.reader(open('cleaned.csv', 'rU'), delimiter= ";",quotechar='"')
#tokenData = nltk.word_tokenize(str(reader))
#for line in reader:
#for field in line:
#tokens = word_tokenize(field)
#print(tokens)



#print(df.head())
#df.to_csv("tokeniseddata.csv")


