#import csv
import pandas as pd
#import numpy as np
#from pandas import *
#import nltk
#import string
#import re
newcsv_1 = pd.read_csv('recout.csv', sep=';')
#df.info()
#concat(tp, ignore_index=True)
print(newcsv_1.shape)
newcsv_1.dropna()
#newcsv_1.to_csv("cleanedcsv1.csv")
#newcsv_2 = pd.read_csv('cleanedcsv1.csv', encoding='latin1', sep = ',')
#print (newcsv_2.head(10))
#print (newcsv_1.head(2))
print(newcsv_1.shape)
newcsv_1.to_csv("cleaned.csv")

newcsv_1
