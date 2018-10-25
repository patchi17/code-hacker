
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import csv
import pandas as pd
from sklearn.model_selection import validation_curve
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas, xgboost, numpy, textblob, string
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
df1= pd.read_csv('labeled.csv', sep=';', encoding='utf8')
X= df1['desstop']
y= df1['label_SCat']
# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['X1'] = X
trainDF['y1'] = y
#print(tokens)
#print(i)
print(trainDF['X1'])
print(trainDF['y1'])
# split the dataset into training and validation datasets
#X1_train, X1_test, y1_train, y1_test = train_test_split(X,y,test_size=0.5, random_state=1)
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['X1'], trainDF['y1'])

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

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
print (xtrain_tfidf_ngram)
print (xvalid_tfidf_ngram)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', tokenizer=text_process, ngram_range=(4,4), max_features=10000)
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


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    #print (predictions)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    #print (predictions)
    return metrics.accuracy_score(predictions, valid_y)

# Naive Bayes on Count Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
#print ("NB, Count Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
#print ("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
#print ("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
#print ("NB, CharLevel Vectors: ", accuracy)


# Linear Classifier on Count Vectors
#accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
#print ("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
#accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
#print ("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
#accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
#print ("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
#accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
#print ("LR, CharLevel Vectors: ", accuracy)

# SVM on Ngram Level TF IDF Vectors
#accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
#print ("SVM, N-Gram Vectors: ", accuracy)

# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(n_estimators = 1000,max_features=2, random_state=0, n_jobs=-1), xtrain_count, train_y, xvalid_count)
print ("RF, Count Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(n_estimators = 5000,max_features=2, random_state=0, n_jobs=-1), xtrain_tfidf, train_y, xvalid_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)


# Extereme Gradient Boosting on Count Vectors
#accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
#print ("Xgb, Count Vectors: ", accuracy)
#
# Extereme Gradient Boosting on Word Level TF IDF Vectors
#accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
#print ("Xgb, WordLevel TF-IDF: ", accuracy)

# Extereme Gradient Boosting on Character Level TF IDF Vectors
#accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
#print ("Xgb, CharLevel Vectors: ", accuracy)

#pred = classifier.predict(xvalid_tfidf_ngram)
#print (metrics.classification_report(y1_test, pred))
#label_SCat = ['Other','DrugandGuns', 'Nudism']
#plt.figure(figsize=(5, 5))
#sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=label_SCat, yticklabels=label_SCat);
#plt.title('Confusion Matrix of the labels')

#def create_model_architecture(input_size):
    # create input layer
    #input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    #hidden_layer1 = layers.Dense(50, activation="relu")(input_layer)
    #hidden_layer2 = layers.Dense(50, activation="relu")(hidden_layer1)
    #hidden_layer3 = layers.Dense(50, activation="relu")(hidden_layer2)
    # create output layer
    #output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer3)
    
    #classifier = models.Model(inputs = input_layer, outputs = output_layer)
    #classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    #return classifier

#classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
#accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
#print ("NN, Ngram Level TF IDF Vectors",  accuracy)


#param_range = np.arange(1, 1000, 2)
#train_scores, test_scores = validation_curve(RandomForestClassifier(), xtrain_tfidf,train_y,param_name="n_estimators",param_range=param_range, cv=3, scoring="accuracy", n_jobs=-1) 

#train_mean = np.mean(train_scores, axis=1)
#train_std = np.std(train_scores, axis=1)

#test_mean = np.mean(test_scores, axis=1)
#test_std = np.std(test_scores, axis=1)


#plt.plot(param_range, train_mean, label="Training score", color="black")
#plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

#plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
#plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")


#plt.title("Validation Curve With Random Forest")
#plt.xlabel("Number Of Trees")
#plt.ylabel("Accuracy Score")
#plt.tight_layout()
#plt.legend(loc="best")
#plt.show()
