
#------------ import 
from multiprocessing import reduction
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import scipy
from scipy import stats, spatial
from scipy.linalg import qr
from scipy.special import logit, expit
import pandas as pd
from operator import add, length_hint 
import itertools
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, mean_squared_error,precision_recall_fscore_support
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import statsmodels.api as sm

import copy
import inspect
import pickle
from enum import Enum
import seaborn as sns
import umap
import random

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.utils.parametrizations as parametrizations

from typing import Union
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo 
import csv
import datetime
import heapq


class PrepareData(Dataset):

    def __init__(self, Y, X=np.array([]), V=None, weight=None
    ):
        
        # X: n * px
        # Y: n * n_gene
        # weight: n * px
        if X.size == 0:
            self.X = torch.from_numpy(np.array([])).float()
        else:
            self.X = torch.from_numpy(np.asarray(X)).float()
            
        self.Y = torch.from_numpy(np.asarray(Y)).float()
        self.naY_pos = torch.from_numpy(np.asarray(np.isnan(Y)))
        temp_Y = copy.deepcopy(np.asarray(Y))
        temp_Y[np.asarray(np.isnan(Y))] = 0
        self.Y_naAs0 = torch.from_numpy(temp_Y).float()  # need to match output dtype "float32"

        if V is None or V.size == 0:
            self.V = torch.from_numpy(np.array([])).float()
        else:
            self.V = torch.from_numpy(np.asarray(V)).float()



        if weight is None:
            self.weight = torch.from_numpy(np.array([])).float()   # because None cannot be allowed as output
        else:
            self.weight = torch.from_numpy(np.asarray(weight)).float()
            
    def __len__(self):
        return self.Y.size()[0]
    
    # get data from certain areas -- not used
    def __getitem__(self, idx): 

        this_X = self.X[idx,:] if self.X.numpy().size != 0 else self.X
        this_weight = self.weight[idx,] if self.weight.numpy().size !=0 else self.weight
        this_V = self.V[idx,:] if self.V.numpy().size != 0 else self.V

        return (this_X, self.Y[idx,:], self.naY_pos[idx,:], self.Y_naAs0[idx,:], this_V, this_weight, idx)


### model with covariates
# pv: is the number of covariates
# penalty: penalty on the quantiles of Z across centers
class model_ZV(torch.nn.Module):
    

    def __init__(self, n_gene, n_factor, px, n_sub, pv,
    W_orthonormal=True, beta_W_orth=False, gamma_W_orth=False,
    Z_demean_bycenter=False, Z_scale_bycenter=False, Z_orthogonal=False,
    penalty=0, nquantile=4,
    l1_gamma=False, l1_W=False, l1_beta=False, 
    fix_W_at = None, fix_beta_at = None, fix_gamma_at = None):
    
    ## parameters
    # W (n_gene * n_factor)
    # beta (n_gene * px)
    # Z (n_sub * n_factor)
        
        # generate a random number bt 0-1000, together with date to identify model objects
        super().__init__()

        self.datetime = datetime.datetime.now()
    
        self.n_gene = n_gene
        self.n_factor = n_factor
        self.px = px
        self.pv = pv
        self.n_sub = n_sub
        self.W_orthonormal = W_orthonormal
        self.beta_W_orth = beta_W_orth
        self.gamma_W_orth = gamma_W_orth
        self.Z_orthogonal = Z_orthogonal

        self.Z_demean_bycenter = Z_demean_bycenter
        self.Z_scale_bycenter = Z_scale_bycenter
        self.penalty = penalty
        self.nquantile = nquantile

        self.l1_gamma = l1_gamma
        self.l1_W = l1_W
        self.l1_beta = l1_beta


         # define orthogonal W
        if self.W_orthonormal:
            self.W = parametrizations.orthogonal(nn.Linear(self.n_factor, self.n_gene, bias=False)) # loading matrix: utilize nn.linear so that
        else:
            self.W = nn.Linear(self.n_factor, self.n_gene, bias=False)
        # fix W
        if fix_W_at is not None:
            self.W.weight = nn.Parameter(fix_W_at) if torch.is_tensor(fix_W_at) else nn.Parameter(torch.tensor(fix_W_at).float())
            for p in self.W.parameters():
                p.requires_grad=False


        # beta    
        if fix_beta_at is not None:
            self.beta = nn.Parameter(fix_beta_at) if torch.is_tensor(fix_beta_at) else nn.Parameter(torch.tensor(fix_beta_at).float())
            self.beta.requires_grad = False
        else: 
            self.beta = nn.Parameter(torch.rand(self.n_gene, self.px))
        self.beta_init = copy.deepcopy(self.beta)

        # gamma
        if fix_gamma_at is not None:
            self.gamma = nn.Parameter(fix_gamma_at) if torch.is_tensor(fix_gamma_at) else nn.Parameter(torch.tensor(fix_gamma_at).float())
            self.gamma.requires_grad = False
        else:
            if self.pv > 0:
                self.gamma = nn.Parameter(torch.rand(self.n_gene, self.pv))

        # Z
        if Z_orthogonal:
            self.Z = parametrizations.orthogonal(nn.Linear(self.n_factor, self.n_sub, bias=False))
        else:
            self.Z = nn.Parameter(torch.rand(self.n_sub, self.n_factor))
        
        # the parameters would be in the same order in model.parameters() as the registration 
        self.register_parameter("beta", self.beta)  # register to the module

        

    # Y: pooled gene across all studies (n * p)
    # X: pooled covariate (n * px)
    # Z: pooled factor score (n * n_factor)        
    def forward(self, XX, Z, V):
        
        # make beta orthogonal to W
        if self.beta_W_orth:
            # project beta to the orthogonal space of W --- NOTE: this will use all beta so beta != beta_init for any beta then
            beta_orthtoW = self.beta - self.W.weight @ self.W.weight.T @ self.beta
            self.beta_orthtoW = beta_orthtoW
        else:
            beta_orthtoW = self.beta
            self.beta_orthtoW = self.beta   # used for l1 penalty calculation
            

        # make gamma orthogonal to W
        if self.pv > 0:
            if self.gamma_W_orth: 
                # project beta to the orthogonal space of W --- NOTE: this will use all beta so beta != beta_init for any beta then
                gamma_orthtoW = self.gamma - self.W.weight @ self.W.weight.T @ self.gamma
                self.gamma_orthtoW = gamma_orthtoW
            else:
                gamma_orthtoW = self.gamma
                self.gamma_orthtoW = self.gamma   # used for l1 penalty calculation

        # calculate first term (beta X)    
        if self.px > 0:
            term1 = XX @ beta_orthtoW.T
        else:
            term1 = 0

        # calculate second term  (WZ)
        term2 = Z @ self.W.weight.T

        # calculate the third term (Gamma V for covariates)
        term3 = V @ gamma_orthtoW.T if self.pv > 0 else 0

        # predict Y prob
        y_pred = torch.sigmoid(term1 + term2 + term3)
        return y_pred


    # # function to calculate loss
    # y_pred: the forward output, if not provided, need X and Z to run the forward function
    # X, Z: dataset attribute type, needed when penalty > 0 or 
    # weight: is a nn.tensor with the same size as y_pred
    def cal_loss(self, naY_pos, Y_naAs0, y_pred=None, X=None, Z=None, V=None, weight=None, reduction='mean', os=None):
        
        if y_pred is None:
            # run the forward function
            y_pred = self(X, Z, V)

        #  Binary Cross Entropy
        if weight is not None and weight.nelement() != 0:  # since weight is a empty tensor created in prepareData
            criterion = torch.nn.BCELoss(reduction = reduction, weight = weight[~naY_pos]) # 'none':output is same dimension as input then, can handle na afterwards
        else:
            criterion = torch.nn.BCELoss(reduction = reduction) 
        
        loss = criterion(y_pred[~naY_pos], Y_naAs0[~naY_pos])  # previous na will have loss 0, when taking the average 0 matters
        loss0 = loss.detach().numpy()  # this is the loss wo the penalty terms


        # add penalty for the difference in the quantiles of Z across centers
        if self.penalty > 0:
            # calculate the quantiles for each Z by center
            quantile_points = np.arange(1, self.nquantile) / self.nquantile
            quantiles = torch.empty((self.px, self.n_factor, (self.nquantile-1)))
            for i in range(self.px): # range over center
                index = X[:,i] == 1
                for j in range(self.nquantile - 1):
                    quantiles[i,:,j] = torch.quantile(Z[index], quantile_points[j], axis=0)
            # sum the pairwise difference between centers (1vs2, 2vs3, ...)
            l2_sum = torch.tensor([0])
            for i in range(1, self.px):
                l2_sum = l2_sum + torch.square(quantiles[i] - quantiles[i-1]).sum()
        else:
            l2_sum = 0

        ## L1 losses for the parameters (adding penalty terms to L0 loss)
        if self.l1_gamma > 0:
            l1_gamma = torch.nn.L1Loss(reduction='mean')
            l1_gamma_loss = l1_gamma(self.gamma_orthtoW, torch.zeros(self.gamma_orthtoW.size()))
        else:
            l1_gamma_loss = 0

        if self.l1_W > 0:
            l1_W = torch.nn.L1Loss(reduction='mean')
            l1_W_loss = l1_W(self.W.weight, torch.zeros(self.W.weight.size()))
        else:
            l1_W_loss = 0

        if self.l1_beta > 0:
            l1_beta = torch.nn.L1Loss(reduction='mean')
            l1_beta_loss = l1_beta(self.beta_orthtoW, torch.zeros(self.beta_orthtoW.size()))
        else:
            l1_beta_loss = 0

        loss = loss + self.penalty * l2_sum + self.l1_gamma * l1_gamma_loss + self.l1_W * l1_W_loss + self.l1_beta * l1_beta_loss

        return {'loss':loss, 'loss0':loss0}



def train_model(model, dataset, n_epochs, optimizer, mask_ind_object=None, batch_size=2000, 
                early_stop=False, min_delta=0.0001, patience=3,
                trueZW = None, trueBeta = None, trueGamma = None,
                plot_every=20, plot_atEnd=True, print_process=True):

    #  penalty: penalty parameter for difference in Z quantiles across centers
    #  nquantile: the quantile points = 1/this number
    #  "mask_ind_object": the object from the class "create_mask_Y"
    #  "mask_ind"==1 entries will not be used in training; 
    #  "mask1_ind==0" entries will not be used for early stopping; 
    #  "mask2_ind==0" entries will not be used for evaluation; 
    #  trueZW: if not None, will be used to calculate loss on ZW

    if mask_ind_object is not None:
        last_mask1_loss0 = 100  # set to a big one to start the process
        last_mask2_loss0 = 100
        trigger_times = 0       # this is for mask1
        trigger_times_mask2 = 0
        # to keep track of number of loss increases in loss0 to see the performance of early stop
        mask1_loss0_increase = np.array([0])
        mask2_loss0_increase = np.array([0])
        # to keep track of which epoch the loss increase (so as to derive # of increases before reaching the minimum)
        mask1_loss0_increase_epoch = np.array([0])
        mask2_loss0_increase_epoch = np.array([0])

    stop_training = False # used to break the nested training loop


    # record in model whether Y_weight is used
    if dataset.weight.detach().numpy().shape[0] == 0:
        model.weight_Y = False
    else:
        model.weight_Y = True

    
    # make the masked Y "mask_ind" (for the training) to NA in the training dataset ("dataset")
    if mask_ind_object is not None:
        dataset_ori = copy.deepcopy(dataset)
        mask_ind = mask_ind_object.mask_ind.values if isinstance(mask_ind_object.mask_ind, pd.DataFrame) else mask_ind_object.mask_ind
        Y_temp = dataset_ori.Y.numpy().copy()

        Y_temp[mask_ind==1] = np.nan
        dataset = PrepareData(Y=Y_temp, X=dataset_ori.X.numpy(), V=dataset_ori.V.numpy(), weight=dataset_ori.weight.numpy()) # used for training 

        # dataset for mask1 genes (make the other non-masked genes to nan) -- used for early stop
        if hasattr(mask_ind_object, 'mask1_ind') and mask_ind_object.mask1_ind is not None:
            mask1_ind = mask_ind_object.mask1_ind.values if isinstance(mask_ind_object.mask1_ind, pd.DataFrame) else mask_ind_object.mask1_ind
            Y_temp = dataset_ori.Y.numpy().copy()
            Y_temp[mask1_ind==0] = np.nan
            dataset_mask1 = PrepareData(Y=Y_temp, X=dataset_ori.X.numpy(), V=dataset_ori.V.numpy(), weight=dataset_ori.weight.numpy()) 

        # dataset for mask2 genes (make the other non-masked genes to nan) -- used for out of sample evaluation
        if hasattr(mask_ind_object, 'mask2_ind') and mask_ind_object.mask2_ind is not None:
            mask2_ind = mask_ind_object.mask2_ind.values if isinstance(mask_ind_object.mask2_ind, pd.DataFrame) else mask_ind_object.mask2_ind
            Y_temp = dataset_ori.Y.numpy().copy()
            Y_temp[mask2_ind==0] = np.nan
            dataset_mask2 = PrepareData(Y=Y_temp, X=dataset_ori.X.numpy(), V=dataset_ori.V.numpy(), weight=dataset_ori.weight.numpy()) 

    # create empty array to store the losses
    batch_loss0_all = np.empty(0)
    batch_loss_all = np.empty(0)

    mask1_loss0_all = np.empty(0)
    mask1_loss_all = np.empty(0)

    mask2_loss0_all = np.empty(0)
    mask2_loss_all = np.empty(0)

    # create empty array to store the loss in ZW 
    ZWloss_train_all = np.empty(0)
    ZWloss_mask1_all = np.empty(0)
    ZWloss_mask2_all = np.empty(0)

    # create empty array to store the loss in beta
    loss_beta_all = np.empty(0)
    # create empty 2d array to store the loss for each gamma 
    loss_gamma_all = np.empty((0,model.pv))

    
    for epoch in range(n_epochs):

        if stop_training:
            break
        
        ds_batch = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for X, Y, naY_pos, Y_naAs0, V, weight, idx in ds_batch:

            # extract Z 
            if model.Z_orthogonal:
                Z = model.Z.weight.clone()
            else:
                Z = model.Z.clone()

            if model.px == 0 and model.pv == 0:  # don't need to demean
                Z = model.Z
            else:
                if model.Z_demean_bycenter:                                         
                    for i in range(dataset.X.shape[1]):
                        index = dataset.X[:,i]==1
                        ZZ = Z.clone()
                        Z[index,] = ZZ[index,] - torch.mean(ZZ[index,], axis=0)
                else:  # overall demean
                     Z = Z - Z.mean(axis=0)

                if model.Z_scale_bycenter:
                    for i in range(dataset.X.shape[1]):
                        index = dataset.X[:,i]==1
                        Z[index,] = Z[index,] / torch.std(Z[index,], axis=0)


            # loss for the batch
            if type(model).__name__ == 'model_ZV':
                batch_loss_res = model.cal_loss(naY_pos, Y_naAs0, X=X, Z=Z[idx,], V=V, weight=weight, os=os)
            else:
                batch_loss_res = model.cal_loss(naY_pos, Y_naAs0, X=X, Z=Z[idx,], weight=weight)
            batch_loss = batch_loss_res['loss']

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
            # output the batch training loss
            if print_process: 
                print('Epoch [{}/{}], Loss0 & Loss: {:.4f}, {:.4f}'.format(epoch+1, n_epochs, batch_loss_res['loss0'], batch_loss.item()))
            batch_loss0_all = np.append(batch_loss0_all, batch_loss_res['loss0'])
            batch_loss_all = np.append(batch_loss_all, batch_loss.item())

            
            ## within each batch, calculate loss for the mask1 and mask2 entries & plot the loss
            if (mask_ind_object is not None):  # and epoch % 1 == 0

                # calculate loss on mask1 entries
                if hasattr(mask_ind_object, 'mask1_ind') and mask_ind_object.mask1_ind is not None:

                    loss_mask1_res = model.cal_loss(dataset_mask1.naY_pos, dataset_mask1.Y_naAs0, 
                                        X=dataset_mask1.X, Z=Z, V=dataset_mask1.V, weight=weight, os=dataset_mask1.os) if type(model).__name__ == 'model_ZV' else model.cal_loss(dataset_mask1.naY_pos, dataset_mask1.Y_naAs0, 
                                        X=dataset_mask1.X, Z=Z, weight=weight)
                    # print('** MASK1 -- Epoch [{}/{}], Loss0 & Loss: {:.4f}, {:.4f}'.format(epoch+1, n_epochs, loss_mask1_res['loss0'], loss_mask1_res['loss'].item()))
                    mask1_loss0_all = np.append(mask1_loss0_all, loss_mask1_res['loss0'])
                    mask1_loss_all = np.append(mask1_loss_all, loss_mask1_res['loss'].item())

                    ## keep track number of loss increase "mask1_loss0_increase" records the number of increases so far for each epoch
                    current_mask1_loss0 = loss_mask1_res['loss0']
                    if current_mask1_loss0 - last_mask1_loss0 > min_delta:
                        trigger_times += 1
                        mask1_loss0_increase = np.append(mask1_loss0_increase, mask1_loss0_increase[-1]+1)
                        if print_process:
                            print(' ~~~~~ Mask1 Loss0 increase:', trigger_times, ' times ~~~~~~', last_mask1_loss0, current_mask1_loss0)

                        if early_stop and trigger_times > patience:
                            print('Epoch [{}/{}], Loss0 & Loss: {:.4f}, {:.4f}'.format(epoch+1, n_epochs, batch_loss_res['loss0'], batch_loss.item()))
                            if print_process:
                                print(' ~~~~ Early stopping based on Mask1 genes Loss0 ~~~~', 
                                'Current Mask1 gene loss:', current_mask1_loss0, 
                                'Last Mask1 loss:', last_mask1_loss0)
                                stop_training = True
                            break
                    else:
                        mask1_loss0_increase = np.append(mask1_loss0_increase, mask1_loss0_increase[-1])
                    last_mask1_loss0 = current_mask1_loss0  # update the latest loss

                # calculate loss on mask2_ind==1 entries
                if hasattr(mask_ind_object, 'mask2_ind') and mask_ind_object.mask2_ind is not None:
                    loss_mask2_res = model.cal_loss(dataset_mask2.naY_pos, dataset_mask2.Y_naAs0, 
                                        X=dataset_mask2.X, Z=Z, V=dataset_mask2.V, weight=weight) if type(model).__name__ == 'model_ZV' else model.cal_loss(dataset_mask2.naY_pos, dataset_mask2.Y_naAs0, 
                                        X=dataset_mask2.X, Z=Z,                    weight=weight)   
                    # print('** MASK2 -- Epoch [{}/{}], Loss0 & Loss: {:.4f}, {:.4f}'.format(epoch+1, n_epochs, loss_mask2_res['loss0'], loss_mask2_res['loss'].item()))
                    mask2_loss0_all = np.append(mask2_loss0_all, loss_mask2_res['loss0'])
                    mask2_loss_all = np.append(mask2_loss_all, loss_mask2_res['loss'].item())

                    ## keep track number of loss increase -- "mask2_loss0_increase" record the number of increases so far for each epoch
                    current_mask2_loss0 = loss_mask2_res['loss0']
                    if current_mask2_loss0 - last_mask2_loss0 > min_delta:
                        trigger_times_mask2 += 1
                        mask2_loss0_increase = np.append(mask2_loss0_increase, mask2_loss0_increase[-1]+1)
                        print(' ~~~~~ Mask2 Loss0 increase:', trigger_times_mask2, ' times ~~~~~~', last_mask2_loss0, current_mask2_loss0)
                    else:
                        mask2_loss0_increase = np.append(mask2_loss0_increase, mask2_loss0_increase[-1])
                    last_mask2_loss0 = current_mask2_loss0  # update the latest loss


            ## calculate loss of WZ (to see whether early stopping makes sense for propsoed method)
            if trueZW is not None and mask_ind_object is not None:

                if model.Z_orthogonal:
                    ZW_all = (model.Z.weight @ model.W.weight.T).detach().numpy() # note W is updated in this iteration but Z is from the previous 
                else: 
                    ZW_all = (model.Z @ model.W.weight.T).detach().numpy() # note W is updated in this iteration but Z is from the previous iteration

                # loss on training data
                ZWloss_train = cal_rmse(ZW_all[mask_ind_object.mask_ind==0], trueZW[mask_ind_object.mask_ind==0])
                ZWloss_train_all = np.append(ZWloss_train_all, ZWloss_train)


                # calculate loss on mask1 entries
                if hasattr(mask_ind_object, 'mask1_ind') and mask_ind_object.mask1_ind is not None:
                    ZWloss_mask1 = cal_rmse(ZW_all[mask_ind_object.mask1_ind==1], trueZW[mask_ind_object.mask1_ind==1])
                    ZWloss_mask1_all = np.append(ZWloss_mask1_all, ZWloss_mask1)

                # calculate loss on mask2 entries
                if hasattr(mask_ind_object, 'mask2_ind') and mask_ind_object.mask2_ind is not None:
                    ZWloss_mask2 = cal_rmse(ZW_all[mask_ind_object.mask2_ind==1], trueZW[mask_ind_object.mask2_ind==1])
                    ZWloss_mask2_all = np.append(ZWloss_mask2_all, ZWloss_mask2)

            if trueBeta is not None:
                beta_temp = copy.copy(model.beta.detach().numpy())

                loss_beta = cal_rmse(beta_temp, trueBeta)
                loss_beta_all = np.append(loss_beta_all, loss_beta)

            if trueGamma is not None:
                loss_gamma = cal_rmse(model.gamma.detach().numpy(), trueGamma, by_column=True)
                loss_gamma_all = np.vstack((loss_gamma_all, loss_gamma))


            ## plot the loss trend along the training process
            if (mask_ind_object is not None and plot_every is not None and (epoch+1) % plot_every == 0): 
                fig, axs = plt.subplots(nrows=1, ncols=5, figsize = (12,3)) # size for the whole fig not individual one
                axs[0].plot(batch_loss0_all)
                axs[0].set_title("Training batch Loss0")

                axs[1].plot(batch_loss_all)
                axs[1].set_title("Training batch Loss (including penalty)")

                axs[2].plot(mask1_loss0_all)
                axs[2].set_title("MASK 1 Loss0")
                if len(mask1_loss0_increase) > 1:
                    for n in [1, 5, 10]:
                        if len(np.where(mask1_loss0_increase==n)[0] > 0):
                            axs[2].vlines(np.where(mask1_loss0_increase == n)[0][0], np.min(mask1_loss0_all), np.max(mask1_loss0_all), color='red')

                axs[3].plot(mask1_loss_all)
                axs[3].set_title("MASK 1 Loss")
                fig.show()

                axs[4].plot(mask2_loss0_all)
                axs[4].set_title("MASK 2 Loss0")
                if len(mask2_loss0_increase) > 1:
                    for n in [1, 5, 10]:
                        if len(np.where(mask2_loss0_increase==n)[0] > 0):
                            axs[4].vlines(np.where(mask2_loss0_increase == n)[0][0], np.min(mask2_loss0_all), np.max(mask2_loss0_all), color='red')
                plt.show()

                if trueZW is not None:
                    ncol = 5 if trueGamma is not None else 4
                    fig, axs = plt.subplots(nrows=1, ncols=ncol, figsize = (12,3)) # size for the whole fig not individual one
                    axs[0].plot(ZWloss_train_all)
                    axs[0].set_title("ZWloss_train_all")

                    axs[1].plot(ZWloss_mask1_all)
                    axs[1].set_title("ZWloss_mask1_all")

                    axs[2].plot(ZWloss_mask2_all)
                    axs[2].set_title("ZWloss_mask2_all")

                    axs[3].plot(loss_beta_all)
                    axs[3].set_title("Beta_loss")

                    if trueGamma is not None:
                        axs[4].plot(loss_gamma_all)
                        axs[4].set_title("Gamma_loss")
                    plt.show()

    ## in the last step -- to make Z demean (and scale) + beta orth to W  -- after the backward step
    # extract Z 
    if model.Z_orthogonal:
        Z = model.Z.weight.clone()
    else:
        Z = model.Z.clone()

    if model.px > 0 or model.pv > 0:
        if model.Z_demean_bycenter is False:
            Z = nn.Parameter((Z - Z.mean(axis=0)))
        else:  # demean by center
            for i in range(dataset.X.shape[1]):
                index = dataset.X[:,i]==1
                Z[index,] = Z[index,] - torch.mean(Z[index,], axis=0)

            model.Z = nn.Parameter(Z)  # Note: no matter whether Z_orthogonal -- hence cannot run the train_model algorithm on the same model twice
        
        if model.beta_W_orth:
            # model.beta_bf = copy.deepcopy(model.beta)
            model.beta = nn.Parameter(model.beta - model.W.weight @ model.W.weight.T @ model.beta)

        if model.pv > 0 and model.gamma_W_orth:
            # model.gamma_bf = copy.deepcopy(model.gamma)
            model.gamma = nn.Parameter(model.gamma - model.W.weight @ model.W.weight.T @ model.gamma)

        if model.Z_scale_bycenter:
            for i in range(dataset.X.shape[1]):
                index = dataset.X[:,i]==1
                Z[index,] = Z[index,] / torch.std(Z[index,], axis=0)  # for each Z dim
            model.Z = nn.Parameter(Z)

    ## beta_real: beta with those non-updating entries to NA
    beta_update = np.empty_like(model.beta.detach().numpy())
    for j in range(dataset.X.shape[1]):
        beta_update[:,j] = 1 - np.isnan(np.nanmean(dataset.Y[dataset.X[:, j]==1, ], 0))
    beta_real = copy.copy(model.beta.detach().numpy())
    beta_real[beta_update==0] = np.nan


    # plot the loss after all the iterations
    if plot_atEnd:
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize = (16,3.5))
        axs[0].plot(batch_loss0_all)
        axs[0].set_title("Training batch Loss0")

        axs[1].plot(batch_loss_all)
        axs[1].set_title("Training batch Loss (including penalty)")

        axs[2].plot(mask1_loss0_all)
        axs[2].set_title("MASK 1 Loss0")

        axs[3].plot(mask1_loss_all)
        axs[3].set_title("MASK 1 Loss")
        fig.show()

    
    # remove the first entry of "mask1_loss0_increase" and "mask2_loss0_increase" (since first entry was add to make the array start)
    mask1_loss0_increase = np.delete(mask1_loss0_increase, 0)
    mask2_loss0_increase = np.delete(mask2_loss0_increase, 0)

 

    ## derive the epoch number that has the lowest mask1_loss0 + how many times of increase at that epoch
    if mask_ind_object.mask1_percent is not None and len(mask1_loss0_all) > 0:
        min_loss0_mask1_epoch = np.argmin(mask1_loss0_all) + 1  # since epoch starts from 0
        n_increase_bf_min_loss0_mask1 = mask1_loss0_increase[min_loss0_mask1_epoch - 1]
        if trueZW is not None:
            min_ZW_loss0_mask1 = min(ZWloss_mask1_all)
            min_ZW_loss0_mask1_epoch = np.argmin(ZWloss_mask1_all) + 1 
        else:
            min_ZW_loss0_mask1 = min_ZW_loss0_mask1_epoch = None
    else:
        min_loss0_mask1_epoch = n_increase_bf_min_loss0_mask1 = None
        min_ZW_loss0_mask1 = min_ZW_loss0_mask1_epoch = None

    if mask_ind_object.mask2_percent is not None and len(mask2_loss0_all) > 0:
        min_loss0_mask2_epoch = np.argmin(mask2_loss0_all) + 1  # since epoch starts from 0
        n_increase_bf_min_loss0_mask2 = mask1_loss0_increase[min_loss0_mask2_epoch - 1]
    else:
        min_loss0_mask2_epoch = n_increase_bf_min_loss0_mask2 = None

    if trueBeta is not None:
        min_beta_loss = min(loss_beta_all)
        min_beta_loss_epoch = np.argmin(loss_beta_all) + 1  # since epoch starts from 0
    else:
        min_beta_loss = min_beta_loss_epoch = None

    if trueGamma is not None:
        loss_gamma_all_sum = np.sum(loss_gamma_all, axis=1)
        min_gamma_loss = min(loss_gamma_all_sum)
        min_gamma_loss_epoch = np.argmin(loss_gamma_all_sum) + 1  # since epoch starts from 
    else:
        min_gamma_loss = min_gamma_loss_epoch = None

 

    # return the specifications and the losses
    return {'n_epochs':n_epochs, 'optimizer':optimizer, 'batch_size':batch_size,
    'loss0_train':batch_loss0_all, 'loss_train':batch_loss_all, 
    'loss0_mask1':mask1_loss0_all, 'loss_mask1':mask1_loss_all, 
    'loss0_mask2':mask2_loss0_all, 'loss_mask2':mask2_loss_all, 

    'mask1_loss0_increase':mask1_loss0_increase, 'mask2_loss0_increase':mask2_loss0_increase,  # will take too much space
    'min_loss0_mask1_epoch':min_loss0_mask1_epoch, 'n_increase_bf_min_loss0_mask1':n_increase_bf_min_loss0_mask1,
    'min_loss0_mask2_epoch':min_loss0_mask2_epoch, 'n_increase_bf_min_loss0_mask2':n_increase_bf_min_loss0_mask2,

    'min_delta': min_delta,
    'ZWloss_train_all':ZWloss_train_all, 'ZWloss_mask1_all':ZWloss_mask1_all, 'ZWloss_mask2_all':ZWloss_mask2_all, 
    'beta_loss':loss_beta_all, 'gamma_loss':loss_gamma_all, # gamma_loss is a 2d array for each gamma
    'min_beta_loss': min_beta_loss, 'min_beta_loss_epoch':min_beta_loss_epoch,
    'min_gamma_loss': min_gamma_loss, 'min_gamma_loss_epoch': min_gamma_loss_epoch,
    'min_ZW_loss0_mask1': min_ZW_loss0_mask1, 'min_ZW_loss0_mask1_epoch':min_ZW_loss0_mask1_epoch,
    'min_delta':min_delta, 
    'early_stop': early_stop, 'patience':patience, 'optimizer':optimizer,
    'beta_real': beta_real 
    }








## function to plot the loss functions (training, mask1, mask2) from the train_model object in one figure with subplots
# plot_loss_increase: False, or a list of iteration number to check the loss
def plot_losses(train_model_obj, from_iter=0, plot_loss_increase=[1, 10, 20], plot_ZWloss=False): # min_delta=0.0001
        

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize = (12,6))
    axs[0,0].plot(train_model_obj['loss0_train'][from_iter:])
    axs[0,0].set_title("Training batch Loss0")

    axs[1,0].plot(train_model_obj['loss_train'][from_iter:])
    axs[1,0].set_title("Training batch Loss (including penalty)")

    axs[0,1].plot(train_model_obj['loss0_mask1'][from_iter:])
    axs[0,1].set_title("MASK 1 Loss0")
    if plot_loss_increase is not False and len(train_model_obj['mask1_loss0_increase']) > 1:
        for n in plot_loss_increase:
            if n <= train_model_obj['mask1_loss0_increase'][-1]:
                axs[0,1].vlines(np.where(train_model_obj['mask1_loss0_increase'] == n)[0][0] - from_iter, 
                np.min(train_model_obj['loss0_mask1'][from_iter:]), np.max(train_model_obj['loss0_mask1'][from_iter:]), color='red')

    axs[1,1].plot(train_model_obj['loss_mask1'][from_iter:])
    axs[1,1].set_title("MASK 1 Loss")

    axs[0,2].plot(train_model_obj['loss0_mask2'][from_iter:])
    axs[0,2].set_title("MASK 2 Loss0")
    if plot_loss_increase is not False and len(train_model_obj['mask2_loss0_increase']) > 1:
        for n in plot_loss_increase:
            if n <= train_model_obj['mask2_loss0_increase'][-1]:
                axs[0,2].vlines(np.where(train_model_obj['mask2_loss0_increase'] == n)[0][0] - from_iter, 
                np.min(train_model_obj['loss0_mask2'][from_iter:]), np.max(train_model_obj['loss0_mask2'][from_iter:]), color='red')

    axs[1,2].plot(train_model_obj['loss_mask2'][from_iter:])
    axs[1,2].set_title("MASK 2 Loss")
    fig.show()


    if plot_ZWloss:
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize = (10, 2.5)) # size for the whole fig not individual one
        axs[0].plot(train_model_obj['ZWloss_train_all'][from_iter:])
        axs[0].set_title("ZWloss_train_all")

        axs[1].plot(train_model_obj['ZWloss_mask1_all'][from_iter:])
        axs[1].set_title("ZWloss_mask1_all")

        axs[2].plot(train_model_obj['ZWloss_mask2_all'][from_iter:])
        axs[2].set_title("ZWloss_mask2_all")

        axs[3].plot(train_model_obj['beta_loss'][from_iter:])
        axs[3].set_title("Beta_loss")

        if 'gamma_loss' in train_model_obj:
            axs[4].plot(train_model_obj['gamma_loss'][from_iter:])
            axs[4].set_title("gamma_loss")

        plt.show()







## define a function to predict Y and calculate the predicted Y accuracy 

# input: X, Z, obs_Y --- all be tensors
# output: "pred_Yprob_real": is using X and Z only for observed genes; "pred_Yprob_corrected": use one intercept mean and Z to predict
# "pred_Yprob_corrected_real": pred_Yprob_corrected for the observed genes only
# "correct_to_Xcol": None then use the mean beta across centers for the _corrected results, if numeric then use that col of beta 
def pred_Y(model, X, Z, obs_Y, V=None, cutpoint=0.5, correct_to_Xcol=None):
    
    # predict Y using X, Z
    pred_Yprob = model(X, Z, V).detach().numpy() if type(model).__name__ == 'model_ZV' else model(X, Z).detach().numpy() # run the forward function
    pred_Y = 1 * (pred_Yprob > cutpoint)
    obs_Y = np.asarray(obs_Y)
    pred_acc = np.mean((obs_Y == pred_Y)[~np.isnan(obs_Y)])
#     pred_acc_ind = np.nanmean(pred_Y == obs_Y, axis=0)  # accuracy for each Y
    

    obsY_freq1 = np.nanmean(obs_Y, axis=0)
    
    pred_Yprob_real = np.where(np.isnan(obs_Y), np.nan, pred_Yprob).copy()
    pred_Y_real = np.where(np.isnan(obs_Y), np.nan, pred_Y).copy()
    
    
    # predict Y using Z only (eleminate the bias from X) -- NOTE: will add an overall intercept to balance the prevalence --- don't need it since we are now correcting it to the MSK level (reference center)
    if model.beta.shape[1] > 1:   # this is only for wX model
        
        
        # new way to define beta_update --- based on the observed data Y
        real_beta = copy.deepcopy(model.beta.detach().numpy())
        beta_update = np.empty_like(model.beta.detach().numpy())
        for j in range(X.shape[1]):
            beta_update[:,j] = 1 - np.isnan(np.nanmean(obs_Y[X[:, j]==1, ], 0))
        real_beta[beta_update==0] = np.nan

        term2 = Z @ model.W.weight.T
        term3 = V @ model.gamma.T if V is not None and model.pv > 0 else 0

        # use Z (also V if any) only to predict
        pred_Yprob_Zonly = torch.sigmoid(term2 + term3).detach().numpy()
        pred_Y_Zonly = 1 * (pred_Yprob_Zonly > cutpoint)
        pred_acc_Zonly = np.nanmean(pred_Y_Zonly == obs_Y)

        if correct_to_Xcol is None:
            # pred_Yprob_corrected -- use mean beta for each gene and Z to predict
            real_beta_mean = torch.from_numpy(np.nanmean(real_beta, axis=1))
        else:
            real_beta_mean = torch.from_numpy(real_beta[:,correct_to_Xcol])
 
        pred_Yprob_corrected = torch.sigmoid(real_beta_mean + term2).detach().numpy()
     
        pred_Y_corrected = 1 * (pred_Yprob_corrected > cutpoint)
        pred_acc_corrected = np.mean((obs_Y == pred_Y_corrected)[~np.isnan(obs_Y)])
        
        # pred_Yprob_corrected for those observed genes only
        pred_Yprob_corrected_real = np.where(np.isnan(obs_Y), np.nan, pred_Yprob_corrected)
        pred_Y_corrected_real = 1 * (pred_Yprob_corrected_real > cutpoint)
        pred_Y_corrected_real = np.where(np.isnan(obs_Y), np.nan, pred_Y_corrected_real)
        

        # these are used to check the results
        term_sum = real_beta_mean + term2
        term2 = term2.detach().numpy()
 
        
    else:  # this is for no X model (same outputs as using X & Z)
        pred_Yprob_Zonly = None
        pred_Y_Zonly = None
        pred_acc_Zonly = None
        
        pred_Yprob_corrected = None
        pred_Y_corrected = None
        pred_acc_corrected = None
        
        pred_Yprob_corrected_real = None
        pred_Y_corrected_real = None
        
        pred_Yprob_corrected_tomsk = None
        pred_Y_corrected_tomsk = None
        
        real_beta = None
        beta_update = None
        real_beta_mean = None
        term_sum = None
        term2 = None
        beta = None
        
    
    return {'pred_Yprob':pred_Yprob, 'pred_Y':pred_Y,
            'pred_acc':pred_acc, 'obsY_freq1':obsY_freq1,
            'pred_Yprob_real':pred_Yprob_real, 'pred_Y_real':pred_Y_real,
            'pred_Yprob_Zonly':pred_Yprob_Zonly, 'pred_Y_Zonly':pred_Y_Zonly, 'pred_acc_Zonly':pred_acc_Zonly, 
            'pred_Yprob_corrected':pred_Yprob_corrected,  'pred_Y_corrected':pred_Y_corrected, 'pred_acc_corrected':pred_acc_corrected,
            'pred_Yprob_corrected_real':pred_Yprob_corrected_real, 'pred_Y_corrected_real':pred_Y_corrected_real, 
            'real_beta':real_beta, 'beta_update':beta_update, 'real_beta_mean':real_beta_mean,
            'term_sum':term_sum, 'term2': term2,
           }

 

## calculate precision recall of the true binary vector using random prediction (from uniform)
# rep: replicate certain number of times and take the mean to avoid randomness
def cal_random_pr(true_vec, rep=3):

    rand_pred = (np.random.uniform(size = len(true_vec)) > 0.5) * 1
    rand_pr = cal_pr_matrix(rand_pred, true_vec) #[:,4:7]

    if rep > 1:
        for i in range(rep):
            rand_pred = (np.random.uniform(size = len(true_vec)) > 0.5) * 1
            rand_pr_temp = cal_pr_matrix(rand_pred, true_vec) #[:,4:7]
            rand_pr = pd.concat([rand_pr, rand_pr_temp])

    # output a df --- 1 line 8 columns
    return pd.DataFrame(np.array(rand_pr.mean(0)).reshape(-1, rand_pr.shape[1]), columns = rand_pr.columns)




## calculate precision recall 
# df_pred, df_true, two dataframes -- columns are genes, rows are observation index
# genes: can be a subset of index for df_pred and df_true (two dataframes)
# compare_w_random: if True, for each gene, will also calculate the pr under random probabiltiy from U(0,1)
def cal_pr_matrix(df_pred, df_true, genes=None, compare_w_random=False, random_rep=3):
    
    if compare_w_random is not True:
        pr_df = pd.DataFrame(index = genes, columns = ['precision_0', 'recall_0', 'f1_0', 'n_0', 'precision_1', 'recall_1', 'f1_1', 'n_1'])
    else:
        pr_df = pd.DataFrame(index = genes, columns = ['precision_0', 'recall_0', 'f1_0', 'n_0', 'precision_1', 'recall_1', 'f1_1', 'n_1', 'random_precision_1', 'random_recall_1', 'random_f1_1'])
    
    if isinstance(df_true, pd.DataFrame):
    
        if genes is None and isinstance(df_true, pd.DataFrame):
            genes = df_true.columns

        for gene in genes:
            # tease out NA
            y_pred = df_pred[gene]
            y_true = df_true[gene]
            is_na = np.isnan(y_pred + y_true)
            
            if sum(~is_na) > 0:
                pr_cal = precision_recall_fscore_support(y_pred = y_pred[~is_na], y_true = y_true[~is_na])
                pr_cal_array = np.array(pr_cal).reshape(-1, order='F') if len(np.array(pr_cal).reshape(-1, order='F'))==8 else np.append(np.array(pr_cal).reshape(-1, order='F'), ([1,1,1,0]))

                if compare_w_random:
                    random_pr = cal_random_pr(y_true[~is_na], rep = random_rep)
                    pr_cal_array = np.concatenate((pr_cal_array, random_pr[['precision_1', 'recall_1', 'f1_1']].values.reshape(-1)))

                pr_df.loc[gene] = pr_cal_array


    if isinstance(df_true, pd.Series):
        
        is_na = np.isnan(df_pred + df_true)
        
        pr_cal = precision_recall_fscore_support(y_pred = np.asarray(df_pred[~is_na]), y_true = np.asarray(df_true[~is_na]))
        pr_cal_array = np.array(pr_cal).reshape(-1, order='F') if len(np.array(pr_cal).reshape(-1, order='F'))==8 else np.append(np.array(pr_cal).reshape(-1, order='F'), ([1,1,1,0]))

        pr_df = pd.DataFrame(pr_cal_array.reshape(-1, len(pr_cal_array)), columns = ["precision_0", "recall_0", "f1_0", "n_0", "precision_1", "recall_1", "f1_1", "n_1"])
        
        if compare_w_random:
            random_pr = cal_random_pr(df_true[~is_na], rep = random_rep)
            pr_df = pd.concat([pr_df, random_pr])

    return pr_df


## function to calculate accuracy & Precision Recall
# Note when creating dataframe, we can remove original index so it will be the natural order, e.g., Y_msk.copy().reset_index(drop=T)
def cal_accuracy_pr(df_pred, df_true, genes=None, comparePR_w_random=False, random_pr_rep=3):
    
    if not isinstance(df_pred, pd.DataFrame):
        df_pred = pd.DataFrame(df_pred, columns = df_true.columns, index = df_true.index)


    if genes is not None:
        df_true = df_true[genes]
        df_pred = df_pred[genes]
    else:
        genes = df_true.columns   

    # first filter the NaN out
    is_na = np.isnan(df_true)
    temp_mat = df_pred == df_true
    temp_mat[is_na] = np.nan

    table_accuracy = pd.DataFrame({
    'accuracy': np.mean(temp_mat), 
    'observed_freq': np.mean(df_true, 0), 
    })

    ## calculate precision recall
    pr_matrix = cal_pr_matrix(df_pred, df_true, genes=genes, compare_w_random=comparePR_w_random, random_rep=random_pr_rep)

    return {'accuracy': table_accuracy, 'pr': pr_matrix}




## Define a function to evaluate the model (for training set, mask1, and mask2 datasets) 
# will calculate accuracy, precision-recall, and pr compared to by random probability if cal_pr==True, 
# df_pred: need to be binary prediction dataframe if cal_pr==True
# if cal_pr==False, can plug in "pred_Yprob" instead of "pred_Y" to only compare TBM (mask2 subjects predicted vs observed)
def eval_model(df_pred, df_true, mask_Y_obj, ignore_warning=True, random_pr_rep=3, cal_pr=True):

    if not isinstance(df_pred, pd.DataFrame):
        df_pred = pd.DataFrame(df_pred, columns = df_true.columns, index = df_true.index)

    if ignore_warning is True:
        import warnings
        warnings.filterwarnings('ignore')


    if cal_pr:
        # for the training sample
        train_res = cal_accuracy_pr(df_pred[mask_Y_obj.mask_ind==False], df_true[mask_Y_obj.mask_ind==False], 
                                    comparePR_w_random=True, random_pr_rep=random_pr_rep)
        train_res_table = pd.concat([train_res['accuracy'].iloc[:,[1,0]], train_res['pr'].iloc[:,4:]], axis=1)

        # for mask1 samples
        if hasattr(mask_Y_obj, "mask1_ind") and mask_Y_obj.mask1_ind is not None:
            mask1_res = cal_accuracy_pr(df_pred[mask_Y_obj.mask1_ind==1], df_true[mask_Y_obj.mask1_ind==1], 
                                        comparePR_w_random=True, random_pr_rep=random_pr_rep)
            mask1_res_table = pd.concat([mask1_res['accuracy'].iloc[:,[1,0]], mask1_res['pr'].iloc[:,4:]], axis=1)
        else:
            mask1_res_table = None

        # for mask2 samples
        if mask_Y_obj.mask2_ind is not None:
            mask2_res = cal_accuracy_pr(df_pred[mask_Y_obj.mask2_ind==1], df_true[mask_Y_obj.mask2_ind==1], 
                                        comparePR_w_random=True, random_pr_rep=random_pr_rep)
            mask2_res_table = pd.concat([mask2_res['accuracy'].iloc[:,[1,0]], mask2_res['pr'].iloc[:,4:]], axis=1)
            # drop the rows with all nan
            mask2_res_table.dropna(axis=0, how='all', inplace=True)
        else:
            mask2_res_table = None


        ## for all result table, add a column for f1_1 > radom_f1_1
        for res_table in [train_res_table, mask1_res_table, mask2_res_table]:
            if res_table is not None:
                res_table['f1_1_better'] = res_table['f1_1'] > res_table['random_f1_1']


        ## creat a summary table (mean from each table)
        summary_table = pd.DataFrame(index = ['train', 'mask1', 'mask2'], columns = train_res_table.columns)
        summary_table.iloc[0, ] = train_res_table.mean(0)
        if hasattr(mask_Y_obj, "mask1_ind") and mask_Y_obj.mask1_ind is not None:
            summary_table.iloc[1, ] = mask1_res_table.mean(0) 
        if mask_Y_obj.mask2_ind is not None:
            summary_table.iloc[2, ] = mask2_res_table.mean(0)
    else:
        train_res_table = mask1_res_table = mask2_res_table = summary_table = None

    return {'train':train_res_table, 'mask1':mask1_res_table, 'mask2':mask2_res_table, 'summary':summary_table}



def sigmoid(x):
    return 1 / (1 + np.exp(-x))



## For the input dataset, mask specified genes / random genes by percentage AMONG non-NA entries
#--- create_mask_inds:
# 1) mask2: mask by percentage2 of patients (random on the patient level) for their selected gene/genes (mask2_genes) -- to evaluate the out of sample imputation performance --- this is like out of sample test --- 
#           we will exclude these patients from sampling the mask1 genes
# 2) mask1: randomly mask by percentage2 the entries in the subject*gene matrix (EXCLUDING the mask2 patients) -- this is for early stopping by the imputation capability
# mask_ind: False for training (note na is also included in False)
# all indicator variables are dataframes of the same size as Y_df
#--- create_mask_inds_2 -- see function comment
class create_mask_Y():

    def __init__(self, Y_df, mask1_percent=None, mask1_genes=None, mask2_sub_percent=None, mask2_percent=None, mask2_genes=None, mask1_seed=1, mask2_seed=2):
        self.Y_df = Y_df 
        self.mask1_percent = mask1_percent
        self.mask1_genes = mask1_genes
        self.mask2_sub_percent = mask2_sub_percent
        self.mask2_percent = mask2_percent
        self.mask2_genes = mask2_genes
        self.mask1_seed = mask1_seed
        self.mask2_seed = mask2_seed 

    ## helper function to create the mask indicator matrix:  1 for masked entries, 0 for unmasked, nan for nan in the original data
    # nonmask_index: the index of the Y matrix that we try to mask the genes on, if 'None' then we don't restrict to specific subjects -- since for mask1 we need to limit to those who are not mask2
    # mask_genes: the candidate genes to mask on, will not mask the other genes
    def _create_mask_ind(self, mask_percent, mask_genes=None, nonmask_index=None, nonmask_entry=None, seed=1):

        if mask_percent is None:
            # print("No mask_percent is provided !!")
            return None
        else:
            Y_mask_ind = ~np.isnan(self.Y_df)  # True for non-NA entries

            if mask_genes is not None:
                Y_mask_ind = ~np.isnan(self.Y_df)
                nonmask_genes = Y_mask_ind.columns.difference(mask_genes, sort=False)
                Y_mask_ind.loc[:, nonmask_genes] = False  # non-selected genes is treated with nan for now (so easier for sampling by percentage later)

            if nonmask_index is not None:
                Y_mask_ind.loc[nonmask_index] = False   # non-masked subjects is treated with nan for now (so easier for sampling by percentage later)

            if nonmask_entry is not None:
                Y_mask_ind[nonmask_entry] = False

            # randomly mask AMONG the True (maskable) entries by mask_percent
            np.random.seed(seed)
            Y_mask_ind[Y_mask_ind==True] = np.random.binomial(1, mask_percent, size = Y_mask_ind.shape) + 1   # since False will be converted to float dtypes with value being 0
            Y_mask_ind = Y_mask_ind.replace({2: 1, 1:0, False: 0}) # False: np.nan

            ## do we need the following?
            # make the nonmask genes have indicator value of 0
            if mask_genes is not None:
                Y_mask_ind.loc[:, nonmask_genes] = np.where(~np.isnan(self.Y_df.loc[:, nonmask_genes]), 0, np.nan)

            # make the nonmask subjects have indictor value of 0
            if nonmask_index is not None:
                Y_mask_ind.loc[nonmask_index] = np.where(~np.isnan(self.Y_df.loc[nonmask_index]), 0, np.nan)

            # reassure that the original nan in the data is nan in the indicator dataframe as well
            Y_mask_ind[np.isnan(self.Y_df)] = np.nan

            # return the indicator dataframe
            return Y_mask_ind 


    ## create mask1_ind, mask2_ind, mask_ind (mask_ind is for training: 0 for all training entries)
    def create_mask_inds(self):

        # create mask2_ind
        if self.mask2_sub_percent > 0 and self.mask2_genes is not None:

            # first to randomly select mask2 subjects 
            self.mask2_index = random.sample(list(self.Y_df.index), round(self.mask2_sub_percent * self.Y_df.shape[0]))
            self.mask2_rownumber = list()  # this is for future use out side of the function 
            for item in self.mask2_index:
                self.mask2_rownumber.append(list(self.Y_df.index).index(item))
            self.nonmask2_rownumber = list(set(range(self.Y_df.shape[0])) - set(self.mask2_rownumber))


            # then mask all their selected genes
            self.mask2_ind = pd.DataFrame(0, index=self.Y_df.index, columns=self.Y_df.columns)
            self.mask2_ind.loc[self.mask2_index, self.mask2_genes] = 1  # note need to use index because also need column names
            self.mask2_ind[np.isnan(self.Y_df)] = np.nan

            # create mask1_ind AMONG the non-mask2 entries (previously was subjects)
            if self.mask1_percent > 0:
                self.mask1_ind = self._create_mask_ind(self.mask1_percent, self.mask1_genes, nonmask_index=None, nonmask_entry=self.mask2_ind==1, seed=self.mask1_seed)
            else:
                self.mask1_ind = None

        else: # create mask1_ind AMONG all
            self.mask2_ind = None
            if self.mask1_percent > 0:
                self.mask1_ind = self._create_mask_ind(self.mask1_percent, self.mask1_genes, nonmask_index=None, nonmask_entry=None, seed=self.mask1_seed)
            else:
                self.mask1_ind = None

        # create mask_ind: any mask in mask1 or mask2 (used for training data)
        if self.mask2_ind is None and self.mask1_ind is None:
            self.mask_ind = self._create_mask_ind(mask_percent=0)  # will takes care of the NA values

        if self.mask2_ind is None and self.mask1_ind is not None:
            self.mask_ind = self.mask1_ind

        if self.mask1_ind is None and self.mask2_ind is not None:
            self.mask_ind = self.mask1_ind
        
        if self.mask1_ind is not None and self.mask2_ind is not None:
            self.mask_ind = (self.mask1_ind + self.mask2_ind > 0)


    ## First randomly mask entires among mask2_genes (test set), then among non-mask2 entries (previously was subjects) randomly mask among mask1_genes (used for early stop) 
    # can input "mask2_ind" directly (so that the same test set is used across methods)
    def create_mask_inds_2(self, mask2_ind=None):
    
        # create mask2_ind (will have mask2)
        if mask2_ind is not None:
            self.mask2_ind = mask2_ind # use the input directly
        elif self.mask2_percent > 0:
            self.mask2_ind = self._create_mask_ind(self.mask2_percent, self.mask2_genes, nonmask_index=None, nonmask_entry=None, seed=self.mask2_seed)

        # create mask1_ind AMONG the non-mask2 entries (previously was subjects)
        if hasattr(self, "mask2_ind"):
            if self.mask1_percent > 0:
                self.mask1_ind = self._create_mask_ind(self.mask1_percent, self.mask1_genes, nonmask_index=None, nonmask_entry=self.mask2_ind==1, seed=self.mask1_seed)
                # have both mask2 and mask1
                self.mask_ind = (self.mask1_ind + self.mask2_ind > 0)
            else:
                self.mask1_ind = None
                self.mask_ind = self.mask2_ind
        # no mask2
        else:
            self.mask2_ind = None
            if self.mask1_percent > 0: 
                self.mask1_ind = self._create_mask_ind(self.mask1_percent, self.mask1_genes, nonmask_index=None, nonmask_entry=None, seed=self.mask1_seed)
                self.mask_ind = self.mask1_ind
            else: # NO mask2, NO mask1
                self.mask1_ind = None
                self.mask_ind = None


## function to calculate the class weight for each Y (to balance 0 and 1)
def cal_Y_weight(Y_df):

    Y_weight = pd.DataFrame(index=Y_df.index,columns=Y_df.columns)
    Y_mean = Y_df.mean(0)
    Y_n = (~np.isnan(Y_df)).sum(0) + 0.001  # to avoid 0 denominator
    for i in range(Y_weight.shape[1]):
        Y_weight.iloc[:,i] = ((Y_df.iloc[:,i]==1) * 1 / Y_mean[i] + (Y_df.iloc[:,i]==0) * 1/(1-Y_mean[i])) / Y_n[i] / 2
    
    return Y_weight




# make datetime object as a string output, e.g., "2022-09-05_2204" (go further to seconds since can have same minute outputs)
def datatime_to_string(datetime):
    return str(datetime)[0:10] + '_' + str(datetime)[11:13] + str(datetime)[14:16] + '_' + str(datetime)[17:19] + str(datetime)[20:26] 


### Function to run linear regression and output variance explained etc
def run_ols(outcome, covariates):

    # add intercept to the model
    covariates_wIntercept = sm.add_constant(covariates)
    # set up and run the model
    ols = sm.OLS(outcome, covariates_wIntercept)
    ols_fit = ols.fit()
    # calculate variance explained
    var_explained = 1 - ols_fit.mse_resid / ols_fit.mse_total

    # calculate MSE
    predictions = ols_fit.predict(covariates_wIntercept)
    rmse = mean_squared_error(outcome, predictions, squared=False)    

    return {'var_explained':var_explained, 'rmse': rmse, 'pvalues':ols_fit.pvalues[1:], 'model':ols_fit}

     

## calculate root-MSE between two np array 
def cal_rmse(a, b, by_column=False):

    if by_column:
        return np.sqrt(np.nanmean(np.square(a - b), axis=0))
    else:
        return np.sqrt(np.nanmean(np.square(a - b)))


 