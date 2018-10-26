
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import csv
import pandas as pd
import pandas, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.tokenize import TreebankWordTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


def text_process(doc):
    punctuations = '''!()-[]{};:'"\,0123456789<>+./?@#$%^&*_~'''
    no_punct = ""
    for senten in doc:
        if senten not in punctuations:
            no_punct = ((no_punct + senten).lower())
    return  (no_punct)
# load the dataset
df1= pd.read_csv('Newdataset.csv', sep=';', encoding='utf8')
X= df1['desstop']
y= df1['label_SCat']
# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['X1'] = X
trainDF['y1'] = y
#print(tokens)
#print(i)

labels = trainDF["y1"].values
print( labels )
onehot_encoder  = preprocessing.OneHotEncoder( sparse=False )
onehot_encoded  = onehot_encoder.fit_transform( labels.reshape(-1, 1) )
print( onehot_encoded )

# split the dataset into training and validation datasets
#X1_train, X1_test, y1_train, y1_test = train_test_split(X,y,test_size=0.5, random_state=1)
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['X1'], onehot_encoded) 

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', tokenizer=text_process)
count_vect.fit(trainDF['X1'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', tokenizer=text_process, max_features=10000)
tfidf_vect.fit(trainDF['X1'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf

tfidf_vect_ngram = TfidfVectorizer(tokenizer=text_process, analyzer='word', ngram_range=(4,4), 
                                                    max_features=10000, use_idf= 'True')
tfidf_vect_ngram.fit(trainDF['X1'])    
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
#print (xtrain_tfidf_ngram)
#print (xvalid_tfidf_ngram)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', tokenizer=text_process, ngram_range=(2,4), max_features=10000)
tfidf_vect_ngram_chars.fit(trainDF['X1'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['X1'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    predictions = predictions.argmax(axis=-1) # convert to "labels" again (0, 1 or 2)
    valid_yy = valid_y.argmax(axis=-1) # convert to "labels" again (0, 1 or 2)
    return metrics.accuracy_score(predictions, valid_yy)

def create_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer1 = layers.Dense(50, activation="relu")(input_layer)
    hidden_layer2 = layers.Dense(50, activation="relu")(hidden_layer1)
    hidden_layer3 = layers.Dense(50, activation="relu")(hidden_layer2)
    # create output layer
    output_layer = layers.Dense(3, activation="sigmoid")(hidden_layer3)
    
    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("NN, Ngram Level TF IDF Vectors",  accuracy)


