import pandas as pd
from urllib import request
from gensim.models import Word2Vec
import numpy as np

def print_recommendations(song_id):
    similar_songs = np.array(model.wv.most_similar(positive=str(song_id), topn=5))[:,0]
    return songs_df.iloc[similar_songs]

data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')

lines = data.read().decode('utf-8').split('\n')[2:]

playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')

songs_file = songs_file.read().decode('utf-8').split('\n')

songs = [s.rstrip().split('\t') for s in songs_file]

songs_df = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])

songs_df = songs_df.set_index('id')

print('Playlist #1:\n', playlists[0], '\n')
print('Playlist #2:\n', playlists[1])

model = Word2Vec(sentences=playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4)

song_id = 2172

print(songs_df.iloc[song_id])

model.wv.most_similar(positive=str(song_id))

print(print_recommendations(song_id))
