# Python 3.7.6
# -*- coding: utf-8 -*-
# Author: Ines Pisetta

import os
import gensim
import numpy as np

extracted_embeddings_path = 'extracted_embeddings/'

if not os.path.exists(extracted_embeddings_path+'tratz/'):
    os.makedirs(extracted_embeddings_path+'tratz')

if not os.path.exists(extracted_embeddings_path+'oseaghdha/'):
    os.makedirs(extracted_embeddings_path+'oseaghdha')

constituents_path = 'constituents/'
constituents = []

transformations_path = 'transformations/tratz/'
if not os.path.exists(transformations_path):
    os.makedirs(transformations_path)


def extract_embeddings(unknown_emb_file, ds):
    
    extract_embeddings.unknown_emb = np.load(unknown_emb_file)
    extract_embeddings.con_vec = {}
    
    if ds == 'oseaghdha':
        # no transformations needed
        
        for word in constituents:
            get_vector_o(word)
    
    elif ds == 'tratz':
        
        with open('constituents/constituents_tratz_transformations.txt', 'r', encoding = 'utf-8') as t:
            extract_embeddings.first_transformation = {line.split(' -> ')[0]:line.split(' -> ')[1] for line in t.read().splitlines()}
        
        extract_embeddings.second_transformation = {}
        extract_embeddings.combined_transformation = {}
        
        for word in constituents:
            get_vector_t(word)
        
        tmp = 'transformations_' + unknown_emb_file.split('/')[1].rsplit('.', 1)[0].replace('_unknown', '') + '.txt'
        
        with open(transformations_path+tmp, 'w', encoding = 'utf-8') as t:
            for k,v in extract_embeddings.combined_transformation.items():
                t.write(k + ' -> ' + v + '\n')
    
    else:
        print('Error')
    
    tmp = unknown_emb_file.split('/')[1].rsplit('.', 1)[0].replace('_unknown', '') + '.txt'
    output_file = extracted_embeddings_path + ds + '/emb_' + tmp

    with open(output_file, 'w', encoding = 'utf-8') as o:
        for k,v in extract_embeddings.con_vec.items():
            o.write(k + ' ' + str(v) + '\n')
    
    with open(output_file.replace('.txt', '_indices.txt'), 'w', encoding = 'utf-8') as i:
        for count, k in enumerate(extract_embeddings.con_vec.keys()):
            i.write(k + ' ' + str(count) + '\n')
    
    lookuptable = np.array(list(extract_embeddings.con_vec.values()))
    #print('\n' + str(len(constituents)-lookuptable.shape[0]) + ' constituents filtered')
    print(lookuptable.shape)
    #print('\n\n')
    np.save(output_file.replace('.txt', '_vectors.npy'), lookuptable)


def get_vector_o(word):
    
    vector = ''
    
    try:
        vector = vectors[word]
    
    except KeyError:
        #print(word, ' not in vocabulary')
        vector = extract_embeddings.unknown_emb
    
    assert len(vector) == 300
    assert isinstance(vector, np.ndarray)
    extract_embeddings.con_vec[word] = vector
    

def get_vector_t(word):
    
    word_t = word
    
    if word in extract_embeddings.first_transformation:
        word_t = extract_embeddings.first_transformation[word]
    
    if '_' in word_t:
        
        word_f = 0
        
        words = word_t.split('_')
        
        if word_t in extract_embeddings.second_transformation:
            extract_embeddings.combined_transformation[word] = extract_embeddings.second_transformation[word_t]
            #print(word)
            pass
        
        else:
            
            if '-'.join(words) in vectors:
                vector = vectors['-'.join(words)]
                word_f = '-'.join(words)
            
            elif ' '.join(words) in vectors:
                vector = vectors[' '.join(words)]
                word_f = ' '.join(words)
            
            elif word_t in vectors:
                vector = vectors[word_t]
                word_f = word_t
                
            else:
                v = []
                for w in words:
                    # Vektoren der einzelnen Wörter
                    try:
                        v.append(vectors[w])
                    # es sei denn, diese sind unbekannt, dann das entsprechende Embedding verwenden
                    except KeyError:
                        v.append(extract_embeddings.unknown_emb)
                v = np.array(v)
                vector = v.mean(axis=0)
                word_f = '#'.join(words)
                
            assert len(vector) == 300
            assert isinstance(vector, np.ndarray)
            assert isinstance(word_f, str)
            extract_embeddings.second_transformation[word_t] = word_f
            extract_embeddings.combined_transformation[word] = word_f
            extract_embeddings.con_vec[word_f] = vector
    
    # one-word expressions
    else:
        try:
            vector = vectors[word_t]
        
        except KeyError:
            #print(word_t, ' not in vocabulary')
            vector = extract_embeddings.unknown_emb
        
        assert len(vector) == 300
        assert isinstance(vector, np.ndarray)
        extract_embeddings.combined_transformation[word] = word_t
        extract_embeddings.con_vec[word_t] = vector
    

def load_glove(emb_file):
    
    vec = {}
    
    with open(emb_file, 'r', encoding = 'utf-8') as e:
        for line in e:
            line = line.rstrip()
            word = line.split(' ', 1)[0]
            vector_string = line.split(' ', 1)[1]
            vector_list = vector_string.split(' ')
            vector = np.array([float(x) for x in vector_list])
            vec[word] = vector
    
    return vec
    

if __name__ == '__main__':
    
    for ds in ['tratz', 'oseaghdha']:
        with open(constituents_path + 'constituents_' + ds + '.txt', 'r', encoding = 'utf-8') as c:
            constituents = c.read().splitlines()
            """
        vectors = load_glove('embeddings/glove.6B.300d.txt')
        extract_embeddings('unknown_embeddings/glove.6B_unknown.npy', ds)
        vectors = load_glove('embeddings/glove.42B.300d.txt')
        extract_embeddings('unknown_embeddings/glove.42B_unknown.npy', ds)
        vectors = load_glove('embeddings/glove.840B.300d.txt')
        extract_embeddings('unknown_embeddings/glove.840B_unknown.npy', ds)
        """
        vectors = gensim.models.KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
        #extract_embeddings('unknown_embeddings/w2v_unknown.npy', ds)
        extract_embeddings('unknown_embeddings/w2v_unknown_1000.npy', ds)
        