import gensim.downloader as api

model = api.load('glove-wiki-gigaword-50')  # Check available models

model.most_similar(model['king'], topn=11)  # Find most similar words to 'king'
