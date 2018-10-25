from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
file = open("18_file.txt", 'rt')
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vector = vectorizer.fit_transform(file)
# summarize
#vector = vectorizer.vocabulary_
print(vectorizer.vocabulary_)
# encode document
#vector = vectorizer.transform(file)

# summarize encoded vector
print(vector.shape)
print(type(vector))
w = vectorizer.vocabulary_
r = vector.toarray()
#appendFile = open(str(idx) + "_file.txt", 'a')
appendFile = open("vector18.txt", 'a')
appendFile.write(str(r))
appendFile.write(str(w))
#appendFile.write(str(row))
#appendFile.write(r)
appendFile.close()
