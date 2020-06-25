# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 00:37:46 2020

@author: Seyni DIOP
"""

#%%  IMPORT PACKAGES

import sys
import numpy as np
import scipy.io
from tqdm import tqdm
from math import ceil
from scipy.special import expit
from sklearn.metrics import mean_squared_error

# %% rbm_class
##Let's define RBM class which will initilize parameters
class init_RBM():
    def __init__(self, p, q):
        '''
        Parameters
        ----------
        p : INT - NUMBER OF VISIBLE UNITS
        q : INT - NUMBER OF HIDDEN UNITS
        Returns
        -------
        None.

        '''
        self.w = np.random.normal(loc = 0, scale = 0.1, size = (p,q))
        self.a = np.zeros(p, dtype=np.float32)
        self.b = np.zeros(q, dtype=np.float32)

# %% lire_alpha_digit
# Define the function lire_alpha_digits

def lire_alpha_digit(filename, list_char=None):
    '''
    Parameters
    ----------
    filename : FILEPATH
    list_char : LIST OR NP.ARRAY OF CARACTERS

    Returns
    -------
    X : NP.ARRAY
        SHAPE=(LENGTH(list_char),20*16)

    '''
    # load file
    data = scipy.io.loadmat(filename)
    if list_char:
        # Retrieve index of the given caracters    
        flatten_alphadigs = [char[0] for char in data['classlabels'][0]]
        char_idxs = [flatten_alphadigs.index(char) for char in list_char]
    else:
        char_idxs = None
        
    X = data['dat'][char_idxs,:] 
    X = np.stack(X.ravel()).reshape(39*len(char_idxs), 20*16).astype(dtype=np.int8)
    return X

#%% entree_sortie_RBM
def entree_sortie_RBM(rbm, inputs):
    '''
    Parameters
    ----------
    rbm : init_RBM CLASS

    Returns
    -------
    NUMPY ARRAY.

    '''
    X = np.dot(inputs, rbm.w) + rbm.b

    return expit(X)

#%% sortie_entree_RBM
def sortie_entree_RBM(rbm, outputs):
    '''
   
    Parameters
    ----------
    rbm : init_RBM CLASS
    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    H = np.dot(outputs, rbm.w.transpose()) + rbm.a
    return expit(H)  

#%% train RBM
    
def train_RBM(rbm, data, epochs=1, lr=0.1, batch_size=32, verbose=True):
        nb_batches_by_epoch = ceil(len(data)/batch_size)
        (p, q)= rbm.w.shape
        idx = np.arange(0, len(data))
        for epoch in range(epochs):
            np.random.shuffle(idx)
            batches_idxs = np.array_split(idx, nb_batches_by_epoch)
            if verbose:
                progress_bar = tqdm(total=nb_batches_by_epoch,file=sys.stdout,position=0)
                progress_bar.set_description(f'Epoch '+ format(epoch,'d'))
            for idx_batch, idxs in enumerate(batches_idxs):
                v0 = data[idxs]
                n = len(v0)
                
                outputs =  entree_sortie_RBM(rbm,v0)
                h0 = (np.random.rand(v0.shape[0], q) < outputs) 
                
                inputs = sortie_entree_RBM(rbm,h0)
                v1 = (np.random.rand(v0.shape[0], p) < inputs)
                
                da = (v0 - v1).mean(axis=0)
                db = (outputs - entree_sortie_RBM(rbm,v1)).mean(axis=0)
                dw = (np.dot(np.transpose(v0), outputs) - np.dot(np.transpose(v1), entree_sortie_RBM(rbm,v1)))/n
                
                rbm.w = rbm.w + lr*dw
                rbm.a = rbm.a + lr*da
                rbm.b = rbm.b + lr*db
                
                #if verbose:
                #    progress_bar.update()
                #    progress_bar.set_description(f'Epoch '+ format(epoch,'d')+' - Batch '+format(idx_batch+1, 'd')+'/'+format(nb_batches_by_epoch,'d'))
            if verbose:
                output = entree_sortie_RBM(rbm, data)
                new_input = sortie_entree_RBM(rbm,output)
                error = mean_squared_error(data, new_input)
                progress_bar.set_postfix({'Error':format(error,'.4f')})
                progress_bar.close()
            
        return rbm

#%% generer_image_RBM

def generer_image_RBM(rbm, image_shape, nb_iter_gibbs, nb_images):
    p, q = rbm.w.shape
    images = []
    for i in range(nb_images):
        x = (np.random.rand(1,p) < 0.5).astype(int)
        for _ in range(nb_iter_gibbs):
            h = (np.random.rand(1,q) < entree_sortie_RBM(rbm,x)).astype(int)
            x = (np.random.rand(1,p) < sortie_entree_RBM(rbm,h)).astype(int)
        images.append(np.reshape(x, image_shape))
    return images
#%% Example RBM

