# Python 3.7.6
# -*- coding: utf-8 -*-
# Author: Ines Pisetta

import os
import numpy as np
import gensim

unknown_embeddings_path = 'unknown_embeddings/'

if not os.path.exists(unknown_embeddings_path):
    os.makedirs(unknown_embeddings_path)

# this function returns an average embedding over all embeddings in the pretrained Glove model
def get_unknown_emb_glove(file):

    vectors = []

    with open(file, 'r', encoding = 'utf-8') as g:
        for line in g:
            line = line.rstrip()
            vector_string = line.split(' ', 1)[1]
            vector_list = vector_string.split(' ')
            vector = np.array([float(x) for x in vector_list])
            vectors.append(vector)

    vectors = np.array(vectors)
    average_vector = np.mean(vectors, axis=0)
    
    print('average of ' + file.split('/')[1])
    print(len(average_vector))
    print(average_vector)
    
    output_file = unknown_embeddings_path + file.split('/')[1].split('.300d.')[0] + '_unknown.npy'

    np.save(output_file, average_vector)


# this function returns an average embedding over all embeddings in the pretrained w2v model
def get_unknown_emb_w2v():
    
    vocab = list(w2v.vocab.keys())
    vectors = []
    
    for word in vocab:
        vectors.append(w2v[word])
    
    vectors = np.array(vectors)
    average_vector = np.mean(vectors, axis=0)
    
    print('average of all w2v vectors')
    print(len(average_vector))
    print(average_vector)
    
    np.save('unknown_embeddings/w2v_unknown.npy', average_vector)


# this function returns an average embedding over the embeddings of the least frequent 1000 tokens in the pretrained w2v model
def get_unknown_emb_w2v_1000():
    
    least_frequent = w2v.index2entity[-1000:]
    vectors = []

    for word in least_frequent:
        vectors.append(w2v[word])

    vectors = np.array(vectors)
    average_vector = np.mean(vectors, axis=0)
    
    print('average of 1000 least frequent from w2v')
    print(len(average_vector))
    print(average_vector)

    np.save('unknown_embeddings/w2v_unknown_1000.npy', average_vector)


if __name__ == '__main__':
    get_unknown_emb_glove('embeddings/glove.6B.300d.txt')
    get_unknown_emb_glove('embeddings/glove.42B.300d.txt')
    get_unknown_emb_glove('embeddings/glove.840B.300d.txt')
    w2v = gensim.models.KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    get_unknown_emb_w2v()
    get_unknown_emb_w2v_1000()