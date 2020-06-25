# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:26:36 2020

@author: Seyni DIOP
"""
import numpy as np
import matplotlib.pyplot as plt
from rbm_functions import lire_alpha_digit,train_RBM, init_RBM,entree_sortie_RBM,sortie_entree_RBM

#%%

def init_DBN(cells):
    '''
    Parameters
    ----------
    cells : NUMPY ARRAY
        DESCRIPTION.

    Returns
    -------
    dbn : LIST
        LIST OF STACKED RBM.

    '''
    dbn = []
    nb_layers = len(cells)
    for i in range(nb_layers):
        dbn.append(init_RBM(cells[i][0], cells[i][1]))
    return dbn


def train_DBN(dbn, data, nb_iterations, lr, batch_size, verbose=False):
    dnn = []
    inputs = data.copy()
    for i in range(len(dbn)):
        if verbose:
            print('\n-------\nLayer',i,'\n-------')
        temp = train_RBM(dbn[i], inputs, nb_iterations, lr, batch_size, verbose=verbose)
        dnn.append(temp)
        inputs = entree_sortie_RBM(dnn[-1], inputs)
        
    return dnn

#%% generer_image_DBN
def generer_image_DBN(dnn, image_shape, nb_iter_gibbs, nb_images):
    images = []
    for i in range(nb_images):
        rand_vector = np.random.rand(1, dnn[0].w.shape[0])
        x = (rand_vector < 0.5)
        for _ in range(nb_iter_gibbs):
            
            # forward pass
            output_vector = entree_sortie_RBM(dnn[0], x)
            rand_vector = np.random.rand(output_vector.shape[0], output_vector.shape[1])
            h = (rand_vector < output_vector)
            for layer in range(1, len(dnn),1):
                output_vector = entree_sortie_RBM(dnn[layer], h)
                rand_vector = np.random.rand(output_vector.shape[0], output_vector.shape[1])
                h = (rand_vector < output_vector)
            
            #h = (rand_vector < h)
            
            # backward pass
            input_vector = sortie_entree_RBM(dnn[-1], h)
            rand_vector = np.random.rand(input_vector.shape[0], input_vector.shape[1])
            x = (rand_vector < input_vector)
            for layer in range(len(dnn)-2, -1, -1):
                input_vector = sortie_entree_RBM(dnn[layer],x)
                rand_vector = np.random.rand(input_vector.shape[0], input_vector.shape[1])
                x = (rand_vector < input_vector)
                
        # reshape image
        images.append(np.reshape(x, image_shape))  
    return images

#%% DBN_example_test

