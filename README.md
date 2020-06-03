# anci_nn
Further experiments on Automatic Noun Compound Interpretation using Neural Networks

This repository contains the code and results of a university project from summer 2020.
Goal was to re-implement the neural network based approach on automatic noun compound interpretation of Dima et al. 2015 and to expand on it in a few ways. The re-implementation was successful and some missing experiments of the original approach have been included, too. Also, the method has been tested on another dataset as well as the dataset used in the approach of Dima et al. and performance has been compared to the original.
The datasets used were the Tratz dataset of Tratz et al. 2011 that contains 19158 noun compounds with 37 relations and the 1443 compound dataset of Ó Séaghdha 2008 with 1443 compounds and 6 relations. The data is not included in this repository. To retrieve it, use the references below.

## Approach
Using neural networks and several different models, the goal was to predict the semantic relation between two given compounds.

The models are based on pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) and [word2vec](https://code.google.com/archive/p/word2vec/) embeddings. The embeddings are also not included in the repository due to their size and because they can easily be retrieved using the given links.

The basic models for the experiments were GloVe 6B, GloVe 42B, GloVe 840B, w2v and w2v_1000. Further models are a model with random embeddings and two combined models: GloVe 6B + w2v_1000 and GloVe 840B + w2v_1000. With the code in this repository one can also conduct further experiments using any combination of embeddings by running network.py which allows for easy-to-use training and testing on both datasets.

Also, the results can be reproduced since a set seed was used while training the models. Model performance is reported in log files.

Details on the implementation can be found in the paper of Dima et al. 2015. We chose a different early stopping criterion and refrained from adjusting the learning rate during training as it didn't have a noticable impact on performance.

## Structure
* constituents: Contains all constituents of both datasets. Due to the Tratz dataset containing multi-word constituents and constituents that only differ in spelling, the constituents had to be mapped to a truly unique set. The unified version of the constituents is included here as well as the mapping for all 140 multi-word constituents.
* data: Contains the data of both datasets splitted in train, val and test sets. However, this is not the original data but a purely numerical representation that serves as the input for the network. A datapoint consists of an index for each of the two constituents and an index for the relation.
* dataset: not included; put the datasets here (see instructions for extract_constituents.py)
* embeddings: not included; put the pre-trained embeddings here (see instructions for get_unknown_embeddings.py)
* extracted_embeddings: Contains the mapping of constituents to indices as well as the constituent dictionary containing indices and the corresponding embeddings that are also used as input for the network.
* logs: Contains the log files that report model performance of all used models on all datasets during training and testing, including information about loss and accuracy on every epoch. Also contains two log files for a small test to compare the performance and convergence time of a model when using learning rate adjustment and when retaining a set learning rate.
* models: Contains the trained models (all specified above). Necessary if you want to perform testing with the network.
* transformations: Contains intermediate data necessary for one of the preprocessing scripts.
* unknown_embeddings: Contains the embedding vectors that have been calculated to represent words that are not in the vocabulary of the pre-trained word embeddings.

## Dependencies
* Python (3.7.6)
* Numpy (1.18.1)
* Gensim (3.8.0)
* Pytorch (1.2.0)

## Code
Input lists the directories that contain the necessary files for the scripts
Output lists the directories that contain the resulting files from the script
The structure of the repository including all directory is crucial for the scripts to work properly.

### Preprocessing
1. extract_constituens.py - extracts the set of all constituents from the datasets
Input: dataset (containing the Tratz and Ó Séaghdha datasets which you need to retrieve separately; they need to be stoed in the following way: 'dataset/Tratz2011_Dataset/Data/tratz2011_fine_grained_random' and 'dataset/1443_Compounds/1443_Compounds.txt')
Output: constituents
2. unify_constituents.py - unifies the set of constituents by mapping multi-word constituents and duplicates to unique representations
Input: constituents
Output: constituents
3. get_unknown_embeddings.py - calculates the embeddings for unknown words
Input: embeddings (containing pre-trained embeddings which you need to download separately)
Output: unknown_embeddings
4. extract_embeddings.py - extracts a constituent dictionary containing the indices of all constituents with their corresponding word embedding; the numpy files are the base for the lookup table in the models
Input: constituents, embeddings, unknown_embeddings
Output: extracted_embeddings, transformations
5. prepare_data.py - converts the data to a numerical representation that serves as input for the network
Input: dataset, extracted_embeddings, transformations
Output: data

### Models/Experiments
network.py - is used to train and test models using any combination of embeddings and any dataset; you can easily run it from the terminal after downloading the repository and installing the dependencies, the script will guide you through the process of chosing a dataset, model, performing fine-tuning, using random embeddings, training and testing
Input: data, models, numpy files from extracted_embeddings
Output: models (saving models during training), stdout (reports performance)

## References
* Dima, Corina. “On the compositionality and semantic interpretation of English noun compounds”. In: Proceedings of the 1st Workshop on Representation Learning for NLP. 2016, pp. 27–39.
* Dima, Corina and Erhard Hinrichs. “Automatic noun compound interpretation using deep neural networks and word embeddings”. In: Proceedings of the 11th International Conference on Computational Semantics. 2015, pp. 173–183.
* Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. “Distributed representations of words and phrases and their compositionality”. In: Advances in neural information processing systems. 2013, pp. 3111–3119.
* Ó Séaghdha, Diarmuid. Learning compound noun semantics. Tech. rep. University of Cambridge, Computer Laboratory, 2008.
* Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. “GloVe: Global Vectors for Word Representation”. In: Empirical Methods in Natural Language Processing (EMNLP). 2014, pp. 1532–1543. url: http://www.aclweb.org/anthology/D14-1162.
* Tratz, Stephen. Semantically-enriched parsing for natural language understanding. University of Southern California, 2011.
* Tratz, Stephen and Eduard Hovy. “A taxonomy, dataset, and classifier for automatic noun compound interpretation”. In: Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics. 2010, pp. 678–687.

## License
I do not own the datasets and the pre-trained embeddings, thus they are not included in this repository.
The idea for the method is intellectual property of Dima et al. 2015.
Any (python) code and log files are my own product and are licensed under the [MIT license](https://github.com/inlpi/telicity_analysis/blob/master/LICENSE.md).
