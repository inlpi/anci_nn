# Python 3.7.6
# -*- coding: utf-8 -*-
# Author: Ines Pisetta

import os

def unify_constituents(constituent_file):
    
    with open(constituent_file, 'r', encoding = 'utf-8') as c:
        constituents = c.read().splitlines()
    
    # all one-word expressions
    unified_constituents = [x for x in constituents if '-' not in x and ' ' not in x]
    multiwords = [x for x in constituents if '-' in x or ' ' in x]
    
    transformations = {}
    
    for el in multiwords:
        
        words = []
        joined = ''
        
        if '-' in el:
            words = el.split('-')
        
        elif ' ' in el:
            words = el.split(' ')
        
        else:
            print('missing case: ', el)
        
        word = ''.join(words)
        joined = '_'.join(words)
        
        if word in unified_constituents:
            transformations[el] = word
        else:
            unified_constituents.append(joined)
            transformations[el] = joined
    
    unified_constituents = [x + '\n' for x in set(unified_constituents)]
    
    with open(constituent_file.replace('.txt', '_transformations.txt'), 'w', encoding = 'utf-8') as t:
        for k,v in transformations.items():
            t.write(k + ' -> ' + v + '\n')
    
    with open(constituent_file.replace('.txt', '_unified.txt'), 'w', encoding = 'utf-8') as u:
        u.writelines(unified_constituents)
        
        
if __name__ == '__main__':
    os.chdir('constituents/')
    unify_constituents('constituents_tratz.txt')
    # OSeaghdha already has unified constituents
    # unify_constituents('constituents_oseaghdha.txt')