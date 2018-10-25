from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
#file = open("testoutput.txt", 'rt')
tfifvec = []
for f in ["18_file.txt", "6_file.txt"]:
    tmp = open(f, "rt").readlines()
    tfifvec = tfifvec + tmp
print(tfifvec)
# create the transform
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
# tokenize and build vocab
vector = vectorizer.fit(tfifvec)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document

#vector = vectorizer.transform(file)
vector = vectorizer.transform(tfifvec)
# summarize encoded vector
print(vector.shape)
print(type(vector))
w = vector.toarray()

#appendFile = open(str(w) + "_file.txt", 'a')
appendFile = open("unigram618.txt", 'a')
appendFile.write(str(w))
#appendFile.write(str(row))
#appendFile.write(r)
appendFile.close()
#max_df=1.0, min_df=1, max_features=20, analyzer='word',
