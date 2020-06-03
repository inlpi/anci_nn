# Python 3.7.6
# -*- coding: utf-8 -*-
# Author: Ines Pisetta

import os

constituents_path = 'constituents/'

if not os.path.exists(constituents_path):
    os.makedirs(constituents_path)

dataset_path_tratz = 'dataset/Tratz2011_Dataset/Data/tratz2011_fine_grained_random'

dataset_file_oseaghdha = 'dataset/1443_Compounds/1443_Compounds.txt'


def extract_constituents_tratz():
    
    files = [dataset_path_tratz + '/' + f for f in os.listdir(dataset_path_tratz)]
    data = []
    
    for file in files:
        if '.tsv' in file:
            with open(file, 'r', encoding = 'utf-8') as d:
                data += d.readlines()
    
    constituents = []
    
    for compound in data:
        splitted = compound.split('\t')
        first_constituent = splitted[0].lower()
        second_constituent = splitted[1].lower()
        constituents.extend([first_constituent, second_constituent])
    
    unique_constituents = list(set(constituents))
    unique_constituents = [line + '\n' for line in unique_constituents]
    
    print('number of unique constituents: ' + str(len(unique_constituents)))
    
    with open(constituents_path + 'constituents_tratz.txt', 'w', encoding = 'utf-8') as c:
        c.writelines(unique_constituents)


def extract_constituents_oseaghdha():
    
    with open(dataset_file_oseaghdha, 'r', encoding = 'utf-8') as d:
        data = d.readlines()
    
    constituents = []
    
    for compound in data:
        splitted = compound.split(' ')
        first_constituent = splitted[0].lower()
        second_constituent = splitted[1].lower()
        constituents.extend([first_constituent, second_constituent])
    
    unique_constituents = list(set(constituents))
    unique_constituents = [line + '\n' for line in unique_constituents]
    
    print('number of unique constituents: ' + str(len(unique_constituents)))
    
    with open(constituents_path + 'constituents_oseaghdha.txt', 'w', encoding = 'utf-8') as c:
        c.writelines(unique_constituents)

   
if __name__ == '__main__':
    extract_constituents_tratz()
    extract_constituents_oseaghdha()