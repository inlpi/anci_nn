# Python 3.7.6
# -*- coding: utf-8 -*-
# Author: Ines Pisetta

import os
import torch
from torch import nn
import numpy as np

np.random.seed(334)
torch.manual_seed(334)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, d_set):
        
        if dataset not in ['tratz', 'oseaghdha'] or d_set not in ['train', 'val', 'test']:
            raise ValueError('loaded data not valid')
        
        out_dim = ''
        if dataset == 'tratz':
            out_dim = 37
            
        elif dataset == 'oseaghdha':
            out_dim = 6
        else:
            raise ValueError('wrong dataset')
            
        data_path = 'data/' + dataset + '/' + d_set + '.txt'
        
        print(data_path)
        
        with open(data_path, 'r', encoding = 'utf-8') as d:
            data_content = d.read().splitlines()
        
        self.data = []
        self.labels = []
        
        for line in data_content:
            spl = line.split(' ')
            fc = int(spl[0])
            sc = int(spl[1])
            lbl = int(spl[2])
            
            self.data.append(torch.LongTensor([fc, sc]))
            self.labels.append(torch.LongTensor([lbl]))
        
    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.data)
    
    def __getitem__(self, index):
        
        X = self.data[index]
        y = self.labels[index]

        return X, y
        
        
class Network(nn.Module):

    def __init__(self, out_dim, emb_file, emb, finetune, random):
    
        super(Network, self).__init__()
        
        embeddings = 'not defined'
        
        if len(emb) == 0:
            pass
        
        elif len(emb) == 1:
            
            emb_file += 'emb_' + emb[0] + '_vectors.npy'
            
            # load pre-trained embeddings for constituents
            embeddings = torch.from_numpy(np.load(emb_file)).float()
        
        else:
            
            embeddings = []
            
            for emb_f in emb:
                
                emb_tmp = emb_file + 'emb_' + emb_f + '_vectors.npy'
                embeddings.append(torch.from_numpy(np.load(emb_tmp)).float())
            embeddings = torch.cat(embeddings, 1)
        
        # Embedding layer shape is |C|*N, where C are the constituents (5231) and N is the embedding size; embedding size/dimensionality is 300 for each word2vec and Glove and greater if a combination of embeddings is used
        self.lookup = 'not defined'
            
        if random:
            if out_dim == 37:
                self.lookup = nn.Embedding(5231,300)
            elif out_dim == 6:
                self.lookup = nn.Embedding(1585,300)
            else:
                print('Error')
        else:
            self.lookup = nn.Embedding.from_pretrained(embeddings)
        
        
        if not finetune:
            print('fine-tuning off')
            self.lookup.require_grad=False
        else:
            print('fine-tuning on')
            self.lookup.require_grad=True
        
        n = ''
        
        if type(embeddings) == str:
            n = 300
        else:
            n = embeddings.shape[1]
        
        #self.norm1 = nn.LayerNorm((n,))
        #self.norm2 = nn.LayerNorm((out_dim,))
        
        # concatenated embedding is size 2xN, hidden layer size = N
        self.hidden = nn.Linear(2*n,n)
        
        # output layer takes 300 dimensional input from hidden layer and maps it to 37 possible relations for tratz or 6 possible for oseaghdha (k)
        self.output = nn.Linear(n,out_dim)
    
    def forward(self, x):
        
        # input is a tensor with the indices to look up
        x = self.lookup(x)
        
        # concatenation of embedding for first and second constituent, creating a tensor of length 2xN
        x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        
        # hidden layer with sigmoid (logistic) activation; Sigmoid is 1/(1+e^(-x))
        x = torch.sigmoid(self.hidden(x))
        
        #x = self.dropout(x)
        
        # output layer with softmax activation; Softmax to choose the relation between the constituents; paper uses softmax, pytorch recommends to use log_softmax in combination with NLLLoss
        x = torch.log_softmax(self.output(x), dim=-1)
        
        return x


def train(dataset, emb=[], finetune=True, random=False):
    
    if len(emb)==0 or random:
        print('using random embeddings')
        random = True
    else:
        print('using pretrained embeddings')
        print(emb)
    
    out_dim = ''
    emb_file = 'extracted_embeddings/'
    
    if dataset == 'tratz':
        out_dim = 37
        
    elif dataset == 'oseaghdha':
        out_dim = 6
    else:
        raise ValueError('wrong dataset')
    
    emb_file += dataset + '/'
    
    model = Network(out_dim, emb_file, emb, finetune, random)
    
    learnables = [p for p in model.parameters()]
    learnables.extend([p for p in model.lookup.parameters()])
    # using SGD with a initial learning rate of 0.9
    optimizer = torch.optim.ASGD(learnables, lr=0.9)
    
    # using Negative Log Likelihood criterion
    criterion = torch.nn.NLLLoss()
    
    trainloader = torch.utils.data.DataLoader(Dataset(dataset, 'train'), batch_size=5, shuffle=False)
    
    testloader = torch.utils.data.DataLoader(Dataset(dataset, 'val'), batch_size=5, shuffle=False)
    
    train_losses, test_losses, test_accuracy = [], [], []
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=1/3, patience=2)
    
    for epoch in range(50):
        
        print('      Epoch: ', epoch+1, ' Learning Rate:     ', optimizer.param_groups[0]['lr'])
        #print('      Epoch: ', epoch+1, ' Momentum:          ', optimizer.param_groups[0]['momentum'])
        
        epoch_loss = 0
        model.train()
        
        for local_batch, local_labels in trainloader:
            
            optimizer.zero_grad()
            
            prediction = model(local_batch)
            # NLLLoss expects (log) probabilities for the prediction and the index of the correct label
            loss = criterion(prediction, local_labels.flatten())
            
            epoch_loss += loss.detach().item()
            
            loss.backward()
            optimizer.step()
        
        epoch_loss = epoch_loss/len(trainloader)
        train_losses.append(epoch_loss)
        print('Train Epoch: ', epoch+1, ' Loss on Train Set: ', epoch_loss)
        
        epoch_loss = 0
        accuracy = 0
        model.eval()
        
        for local_batch, local_labels in testloader:
            
            prediction = model(local_batch)
            # NLLLoss expects (log) probabilities for the prediction and the index of the correct label
            loss = criterion(prediction, local_labels.flatten())
            
            epoch_loss += loss.detach().item()
            
            top_class = torch.argmax(prediction, dim=-1)
            equals = (top_class.flatten() == local_labels.flatten())
            accuracy += float(torch.mean(equals.type(torch.FloatTensor)))
        
        epoch_loss = epoch_loss/len(testloader)
        test_losses.append(epoch_loss)
        print('Val   Epoch: ', epoch+1, ' Loss on Val   Set: ', epoch_loss)
        
        accuracy = accuracy/len(testloader)
        test_accuracy.append(accuracy)
        print('Val   Epoch: ', epoch+1, ' Accuracy Val  Set: ', accuracy, '\n')
        
        #scheduler.step(accuracy)
        
        # change lr if error is lower than a certain threshold; i.e. accuracy is higher (not specified in paper)
        #if accuracy>0.74:
            #optimizer.param_groups[0]['lr'] = 0.3
        
        """
        early stopping, if error on Val increases for 5 successive epochs, i.e. accuracy decreases
        discarded because rarely occuring, need to find another stopping criterion
        if epoch > 5 and test_accuracy[-6:] == sorted(test_accuracy[-6:], reverse=True):
            break
        """
        
        if epoch > 4 and round(test_accuracy[-3],3) == round(test_accuracy[-2],3) == round(test_accuracy[-1],3):
            break
    
    ft = ''
    if finetune:
        ft = 'finetuned/'
    else:
        ft = 'not_finetuned/'
        
    if not os.path.exists(model_path+dataset+'/'+ft):
        os.makedirs(model_path+dataset+'/'+ft)
    
    if not random:
        torch.save(model.state_dict(), model_path+dataset+'/'+ft+'model_'+'_'.join(emb)+'.pt')
    else:
        torch.save(model.state_dict(), model_path+dataset+'/'+ft+'model_rand_emb'+'.pt')
    
    """
    # testing lr decrease on/off: at which epoch does the model stop?
    total_epochs.append(len(test_accuracy))
    total_accuracy.append(test_accuracy[-1])
    """


def test(dataset, emb=[], finetune=True, random=False):
    
    emb_string = ''
    
    if len(emb)==0 or random:
        random = True
        emb_string = 'rand_emb'
        
    else:
        emb_string = '_'.join(emb)
    
    out_dim = ''
    emb_file = 'extracted_embeddings/'
    
    if dataset == 'tratz':
        out_dim = 37
        
    elif dataset == 'oseaghdha':
        out_dim = 6
    else:
        raise ValueError('wrong dataset')
    
    emb_file += dataset + '/'
    
    ft_string = ''
    
    if finetune:
        ft_string = 'finetuned/'
    else:
        ft_string = 'not_finetuned/'
    
    trained_model_path = 'models/'+dataset+'/'+ft_string+'model_'+emb_string+'.pt'
    print(trained_model_path)
    
    model = Network(out_dim, emb_file, emb, finetune, random)
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    
    testloader = torch.utils.data.DataLoader(Dataset(dataset, 'test'), batch_size=5, shuffle=False)
    
    accuracy = 0
    
    for local_batch, local_labels in testloader:
        
        prediction = model(local_batch)
        
        top_class = torch.argmax(prediction, dim=-1)
        equals = (top_class.flatten() == local_labels.flatten())
        accuracy += float(torch.mean(equals.type(torch.FloatTensor)))
    
    accuracy = accuracy/len(testloader)
    print('Accuracy Test Set: ', accuracy, '\n')

    
if __name__ == '__main__':
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    """
    total_epochs = []
    total_accuracy = []
    for i in range(10):
        print(i+1)
        train('tratz', ['w2v_1000'])
    print('on average ', np.mean(np.array(total_epochs)), ' epochs until convergence')
    print('average accuracy ', np.mean(np.array(total_accuracy)), ' at time of convergence')
    """
    inp_dataset = ''
    inp_tr_inf = ''
    inp_rand = ''
    inp_emb = ''
    inp_ft = ''

    inp_dataset = input('Which dataset do you want to use?\nFor Tratz, type: 1\nFor Oseaghdha, type: 2\n')

    inp_tr_inf = input('Do you want to train a model or use an existing model and evaluate it on the test set?\nIf you want to train, type: 1\nIf you want to use an existing model, type: 2\n')

    inp_rand = input('Do you want to use pre-trained embeddings or random embeddings?\nFor pre-trained embeddings, type: 1\nFor random embeddings, type: 2\n')

    if int(inp_rand) == 1:
        
        inp_emb = input('Which embeddings do you want to use?\nYou can choose from: glove.6B, glove.42B, glove.840B, w2v and w2v_1000 or any combination of these.\nNote however, that already trained models are not available for all combinations.\nType the exact name of your chosen embedding. If you want to use multiple, separate them with a comma: glove.6B, w2v\n')
        
    inp_ft = input('Do you want to fine-tune the embeddings or keep them fixed (i.e. exclude them from the backpropagation process)?\nFor fine-tuning, type: 1\nFor fixed embeddings, type:2\n')


    if int(inp_dataset) == 1:
        inp_dataset = 'tratz'
    elif int(inp_dataset) == 2:
        inp_dataset = 'oseaghdha'
    else:
        raise ValueError('Wrong number for dataset')

    if int(inp_tr_inf) == 1:
        inp_tr_inf = 1
    elif int(inp_tr_inf) == 2:
        inp_tr_inf = 2
    else:
        raise ValueError('Wrong number for training/testing')

    if int(inp_rand) == 1:
        inp_rand = False
        for x in inp_emb.split(','):
            if x.strip() not in ['glove.6B', 'glove.42B', 'glove.840B', 'w2v', 'w2v_1000']:
                raise ValueError('Wrong dataset')
        inp_emb = [x.strip() for x in inp_emb.split(',')]
        
    elif int(inp_rand) == 2:
        inp_rand = True
    else:
        raise ValueError('Wrong number for pretrained/random embeddings')
    
    if int(inp_ft) == 1:
        inp_ft = True
    elif int(inp_ft) == 2:
        inp_ft = False
    else:
        raise ValueError('Wrong number for finetuning y/n')
    
    if inp_tr_inf == 1:
        if inp_rand:
            train(inp_dataset, finetune=inp_ft, random=inp_rand)
        else:
            train(inp_dataset, inp_emb, finetune=inp_ft, random=inp_rand)
    elif inp_tr_inf == 2:
        if inp_rand:
            test(inp_dataset, finetune=inp_ft, random=inp_rand)
        else:
            test(inp_dataset, inp_emb, finetune=inp_ft, random=inp_rand)
    else:
        print('Error')
    """
    for ds in ['tratz', 'oseaghdha']:
        for el in [True, False]:
            train(ds, ['glove.6B'], finetune=el)
            train(ds, ['glove.42B'], finetune=el)
            train(ds, ['glove.840B'], finetune=el)
            train(ds, ['w2v'], finetune=el)
            train(ds, ['w2v_1000'], finetune=el)
            train(ds, random=True, finetune=el)
            train(ds, ['glove.6B', 'w2v_1000'], finetune=el)
            train(ds, ['glove.840B', 'w2v_1000'], finetune=el)
    
    for ds in ['tratz', 'oseaghdha']:
        for el in [True, False]:
            test(ds, ['glove.6B'], finetune=el)
            test(ds, ['glove.42B'], finetune=el)
            test(ds, ['glove.840B'], finetune=el)
            test(ds, ['w2v'], finetune=el)
            test(ds, ['w2v_1000'], finetune=el)
            test(ds, random=True, finetune=el)
            test(ds, ['glove.6B', 'w2v_1000'], finetune=el)
            test(ds, ['glove.840B', 'w2v_1000'], finetune=el)
    """