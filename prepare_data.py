# Python 3.7.6
# -*- coding: utf-8 -*-
# Author: Ines Pisetta

import os
import numpy as np
import random

tratz_path = 'dataset/Tratz2011_Dataset/Data/tratz2011_fine_grained_random/'
tratz_embeddings = 'extracted_embeddings/tratz/'
oseaghdha_path = 'dataset/1443_Compounds/'
oseaghdha_embeddings = 'extracted_embeddings/oseaghdha/'

data_p = 'data/'
data_path = ''

with open(tratz_path+'classes.txt', 'r', encoding = 'utf-8') as tc:
    tratz_classes = {cl:idx for idx, cl in enumerate(tc.read().splitlines())}
#print(tratz_classes)

oseaghdha_classes = {'ABOUT':0, 'ACTOR':1, 'BE':2, 'HAVE':3, 'IN':4, 'INST':5}

oseaghdha_file = oseaghdha_path + '1443_Compounds.txt'
with open(oseaghdha_file, 'r', encoding = 'utf-8') as f:
    oseaghdha_data = f.read().splitlines()
    
random.shuffle(oseaghdha_data)
# 1443 splitted in 70% train, 10% val and 20% test
oseaghdha_datasets = {'train':oseaghdha_data[:1010], 'val':oseaghdha_data[1010:1154], 'test':oseaghdha_data[1154:1443]}


def prepare_data_tratz(word_to_idx):
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    tmp = 'transformations/tratz/transformations' + word_to_idx.strip('emb').replace('_indices', '')
    print(tmp)
    
    with open(tmp, 'r', encoding = 'utf-8') as t:
        transformations = {line.split(' -> ')[0]:line.split(' -> ')[1] for line in t.read().splitlines()}
    
    with open(tratz_embeddings+word_to_idx, 'r', encoding = 'utf-8') as e:
        constituent_dict = {line.rsplit(' ', 1)[0]:int(line.rsplit(' ', 1)[1]) for line in e.read().splitlines()}
    #print(constituent_dict)
    
    files = [tratz_path + f for f in os.listdir(tratz_path) if '.tsv' in f]
    
    for file in files:
        
        print(file)
        
        with open(file, 'r', encoding = 'utf-8') as d:
            data = d.read().splitlines()
        
        prepared_data = []
        
        for sample in data:
            fc, sc, label = sample.split('\t')
            label = str(tratz_classes[label])
            datapoint = str(constituent_dict[transformations[fc.lower()]]) + ' ' + str(constituent_dict[transformations[sc.lower()]])
            prepared_data.append(datapoint + ' ' + label + '\n')
            #print(datapoint, label)
        
        output_file = data_path + file.split('.')[0].split('/')[-1] + '.txt'
        
        with open(output_file, 'w', encoding = 'utf-8') as pd:
            pd.writelines(prepared_data)
        
        print(output_file)
        

def prepare_data_oseaghdha(word_to_idx):
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    with open(oseaghdha_embeddings+word_to_idx, 'r', encoding = 'utf-8') as e:
        constituent_dict = {line.rsplit(' ', 1)[0]:int(line.rsplit(' ', 1)[1]) for line in e.read().splitlines()}
    #print(constituent_dict)
    
    for name, dataset in oseaghdha_datasets.items():
        
        prepared_data = []
        
        for sample in dataset:
            fc = sample.split(' ')[0]
            sc = sample.split(' ')[1]
            label = sample.split(' ')[2]
            label = str(oseaghdha_classes[label])
            datapoint = str(constituent_dict[fc.lower()]) + ' ' + str(constituent_dict[sc.lower()])
            prepared_data.append(datapoint + ' ' + label + '\n')
            #print(datapoint, label)
        
        output_file = data_path + name + '.txt'
        
        with open(output_file, 'w', encoding = 'utf-8') as pd:
            pd.writelines(prepared_data)
        
        print(output_file)


if __name__ == '__main__':
    """
    since the indices are the same for each kind of embedding since we have a constituent dictionary of a fixed size (5231)
    we only need one _indices file to convert the data into indices for the constituents and the labels
    thus we only have one train, val and test file for each dataset and we still can use different kinds of embeddings in the network later (and also more easily)
    """
    data_path = data_p + 'tratz/'
    prepare_data_tratz('emb_glove.6B_indices.txt')
    """
    prepare_data_tratz('emb_glove.42B_indices.txt')
    prepare_data_tratz('emb_glove.840B_indices.txt')
    prepare_data_tratz('emb_w2v_indices.txt')
    prepare_data_tratz('emb_w2v_1000_indices.txt')
    """
    
    data_path = data_p + 'oseaghdha/'
    prepare_data_oseaghdha('emb_glove.6B_indices.txt')
    """
    prepare_data_oseaghdha('emb_glove.42B_indices.txt')
    prepare_data_oseaghdha('emb_glove.840B_indices.txt')
    prepare_data_oseaghdha('emb_w2v_indices.txt')
    prepare_data_oseaghdha('emb_w2v_1000_indices.txt')
    """
    