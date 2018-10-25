import gensim
from gensim.models import word2vec

wordembed = open("6_file.txt", "rt").readlines()
embedded = [sentence.split() for sentence in wordembed]
model = word2vec.Word2Vec(embedded, min_count=1)

# we need to pass splitted sentences to the model
#embedded = [sentence.split() for sentence in wordembed]
#model = word2vec.Word2Vec(embedded, size=100, window=5, min_count=5, workers=4)
print(model)
model.save("model")
model = word2vec.Word2Vec.load("model")
#model.most_similar('antique')
print(model['fleamarkets'])
#appendFile = open("testembedd.txt", 'w')
#appendFile.write(str(w))
