# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:49:24 2020

@author: Seyni DIOP
"""
#%% IMPORT PACKAGES
import sys
import numpy as np
from rbm_functions import entree_sortie_RBM
from dbn_functions import train_DBN
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from copy import deepcopy, copy


#%% calcul softmax
def calcul_softmax(rbm, data):
    X = np.dot(data, rbm.w) + rbm.b
    #stabilize softmax
    return softmax(X, axis=1)

#%% entree_sortie_reseau
def entree_sortie_reseau(dnn, data):
    layers_output = []
    # add input data
    layers_output.append(data)
    for dbn in dnn:
        layers_output.append(entree_sortie_RBM(dbn, layers_output[-1]))
        
    proba = calcul_softmax(dnn[-1], layers_output[-2])
    return data, layers_output[1:], proba

#%% retropropagation
    
def retropropagation(dnn, x_train, y_train, epochs=1, lr=0.1, batch_size=32, pretrain=False, epochs_rbm=None, verbose=True):
    assert len(x_train)==len(y_train)
    
    # If pretrain is selected, pretrain model before
    if pretrain:
        print("Pretraining...")
        dnn =  train_DBN(dnn, x_train, epochs_rbm, lr, batch_size, verbose=False)
        print("Pretraining finished")
        
    
    # nb batches
    nb_batches_by_epoch = int(len(y_train)/batch_size)
    # define a history of entropy during training
    history = []
    y_onehot= OneHotEncoder().fit_transform(y_train.reshape(-1,1))
    idx = np.arange(0, len(x_train))
    
    
    # Define list of optimizer for each matrices
    w_optimizer = []
    b_optimizer = []
    for i in range(len(dnn)):
        w_optimizer.append(AdamOptimizer(dnn[i].w, lr=lr))
        b_optimizer.append(AdamOptimizer(dnn[i].b, lr=lr))
    for epoch in range(epochs):
        np.random.shuffle(idx)
        batches_idxs = np.array_split(idx, nb_batches_by_epoch)
        #w_optimizer = 
        if verbose:
            progress_bar = tqdm(total=nb_batches_by_epoch,file=sys.stdout,position=0)
        
        for idx_batch,idxs in enumerate(batches_idxs):
            dnn_copy = deepcopy(dnn)
            n = len(idxs)
            data, layers_output, proba = entree_sortie_reseau(dnn,x_train[idxs])
            
            #derivative of the last layer
            diff_z = proba - y_onehot[idxs]
            diff_a = np.dot(diff_z, dnn[-1].w.transpose())
            
            # calculate gradient
            delta_w = np.dot(layers_output[-2].transpose(), diff_z)/n
            delta_b = np.sum(diff_z, axis=0)
            ## gradient descent
            #dnn_copy[-1].w = dnn_copy[-1].w - lr*delta_w 
            #dnn_copy[-1].b = dnn_copy[-1].b - lr*delta_b
            dnn_copy[-1].w = w_optimizer[-1].backward_pass(delta_w)
            dnn_copy[-1].b = b_optimizer[-1].backward_pass(delta_b)
            
            for layer in range(len(dnn)-2, -1, -1):
                if layer==0:
                    layer_input = x_train[idxs]
                else:
                    layer_input = layers_output[layer-1]
                    
                diff_sigmoid = np.multiply((1 - layers_output[layer]), layers_output[layer])            
                diff_z = np.multiply(diff_a , diff_sigmoid)
                diff_a = np.dot(diff_z, dnn[layer].w.transpose())
                
                # calculate gradient
                delta_w = np.dot(layer_input.transpose(), diff_z)/n
                delta_b = diff_z.sum(axis = 0)/n
                
                dnn_copy[layer].w = w_optimizer[layer].backward_pass(delta_w)
                dnn_copy[layer].b = b_optimizer[layer].backward_pass(delta_b)
                #dnn_copy[layer].w = dnn_copy[layer].b - lr*delta_w
                #dnn_copy[layer].b = dnn_copy[layer].b - lr*delta_b
            dnn = dnn_copy
            if verbose:
                progress_bar.update()
                progress_bar.set_description(f'Epoch '+ format(epoch+1,'d')+'/'+format(epochs,'d')+' - Batch '+format(idx_batch+1, 'd')+'/'+format(nb_batches_by_epoch,'d'))
        
        _, _, proba = entree_sortie_reseau(dnn, x_train)
        log_likelihood = -np.log(proba[range(len(x_train)),y_train])
        entropy_loss = np.sum(log_likelihood) / len(x_train)
        history.append(entropy_loss)
        if verbose:
            progress_bar.set_postfix({'loss':format(history[-1],'.5f')})
            progress_bar.close()
            
    return dnn, history
#%% test_DNN
    
def test_DNN(dnn, x_test, y_test):
    n = len(y_test)
    _, _, pred_proba = entree_sortie_reseau(dnn, x_test)
    log_likelihood = -np.log(pred_proba[range(n),y_test])
    entropy_loss = np.sum(log_likelihood) / n
    #print("Cross_entropy_loss:", entropy_loss)
    # predicted class
    pred_class = np.argmax(pred_proba, axis=1)
    
    # detrmine the accuracy score
    acc = accuracy_score(y_test, pred_class)
    print('\nAccuracy:', acc)
    print('Cross entropy loss:', entropy_loss,'\n')
    # confusion_matrix (after)
    return acc, entropy_loss, pred_proba

# Let's define a function to compare pretrained and not pretrained model.
def train_compare(dnn, x_train, y_train, x_test, y_test,
                 epochs_retro, lr, batch_size,
                 epochs_rbm,
                 verbose=True):
    '''
    Given a dnn plot 
    '''
    dnn_copy = copy(dnn)
    
    # 2-3 Pretain dbn_1 (unsupervised) and  train (supervised) it with retropropagation algo
    print("1- Pretrain+Retropopagate")
    dnn_1, history_1 = retropropagation(dnn, x_train, y_train,
                                  epochs_retro, lr, batch_size,
                                  pretrain=True, epochs_rbm=epochs_rbm,
                                  verbose=verbose)

    # 4. train (supervised) dbn_2 with just retropropagation algo
    print("\n----------------\n2- Only retropropagation\n")
    dnn_2, history_2 = retropropagation(dnn_copy, x_train, y_train,
                                  epochs_retro, lr, batch_size,
                                  pretrain=False, epochs_rbm=epochs_rbm,
                                  verbose=verbose)
    in_train_1 =  test_DNN(dnn_1,x_train,y_train)[0]
    in_train_2 =  test_DNN(dnn_2,x_train,y_train)[0]
    
    
    
    in_test_1 = test_DNN(dnn_1,x_test,y_test)[0]
    in_test_2 = test_DNN(dnn_2,x_test,y_test)[0]
     
    return (in_train_1, in_train_2, in_test_1, in_test_2), (history_1, history_2)

#%%
class AdamOptimizer:
    def __init__(self, weights, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.theta = weights

    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*np.square(gradient)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        self.theta = self.theta - self.lr*(m_hat/(np.sqrt(v_hat) - self.epsilon))
        return self.theta
        
            