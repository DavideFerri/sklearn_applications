#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:24:35 2020

@author: davideferri
"""

import numpy as np 
import pandas as pd
import scipy.stats as ss
from SynthetData_Gen import synthetic_data_generator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

#import matplotlib.pyplot as plt

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------------- configuration ------------------------------ # 

# set the sample size
size = 1000
# set the dgp
dgp = "sigmoid"
# set the parameters
true_parameters = {"alpha" : 2, "beta1" : 2, "beta2" : 4} 
# set the test size as a proportion 
test_size = 0.2


# -------------------- generata synthetic data ---------------------- # 

# produce synthetic data 
data = synthetic_data_generator(size = size,dgp = dgp,parameters = true_parameters,plots = True) 

# -------------------- preprocessing data --------------------------- # 

# get the splin between train and test data
X_train, X_test, y_train, y_test = train_test_split(data[:,1:],data[:,0],test_size = test_size)
log.info("X train shape %s",X_train.shape) ; log.info("X test shape %s",X_test.shape)

# ------------------- specify estimators/models ----------------------------- # 

models = [RandomizedSearchCV(make_pipeline(StandardScaler(),LinearSVC()),param_distributions = {'linearsvc__C': ss.expon(scale=100)},n_iter = 10),
        RandomizedSearchCV(make_pipeline(StandardScaler(),LogisticRegression()),param_distributions = {'logisticregression__C': ss.expon(scale=100)},n_iter = 10),
        RandomizedSearchCV(make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes = 2)),param_distributions = {'mlpclassifier__alpha': ss.expon(scale=100)},n_iter = 10),
        RandomizedSearchCV(MLPClassifier(hidden_layer_sizes = 2),param_distributions = {'alpha': ss.expon(scale=100)},n_iter = 10)]
          
for model in models:
    #print(model.get_params().keys())
    model.fit(X_train,y_train)
    print(model.best_score_)
    print(model.best_params_)
    