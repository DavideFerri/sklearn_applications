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
from sklearn.metrics import accuracy_score

#import matplotlib.pyplot as plt

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------------- utilities --------------------------------- # 

def model_test_validate(model,X_train,y_train,X_test,y_test):
    """
    
    """
    # fit model
    model.fit(X_train,y_train)
    # make predictions
    y_test_pred = model.predict(X_test)
    # get the MSE and MAE
    print(accuracy_score(y_test,y_test_pred))
    # plt results
    _,ax = plt.subplots()
    ax.plot(   X_test[np.where(y_test == y_test_pred),0],
                  X_test[np.where(y_test == y_test_pred),1],
                  ".",color = "green")
    ax.plot(   X_test[np.where(y_test != y_test_pred),0],
                  X_test[np.where(y_test != y_test_pred),1],
                  ".",color = "red")
    ax.set_xlabel("x1") ; ax.set_ylabel("x2")

# --------------------- configuration ------------------------------ # 

# set the sample size
size = 3000
# set the dgp
dgp = "quadratic"
# set the parameters
true_parameters = {"alpha" : 2, "beta1" : 2, "beta2" : 4} 
# set the test size as a proportion 
test_size = 0.2

# ------------------- specify estimators/models ----------------------------- # 

models = [
        # first model - Linear SVC with standard scaler
        RandomizedSearchCV(
                make_pipeline(StandardScaler(),LinearSVC(max_iter = 500)),
                param_distributions = {'linearsvc__C': ss.expon(scale=100)},
                n_iter = 10,
                cv = 5,
                n_jobs = -1),            
        
        #  second model - Logistic regression with standard scaler
        RandomizedSearchCV(
                make_pipeline(StandardScaler(),LogisticRegression(solver = 'lbfgs',max_iter = 500)),
                param_distributions = {'logisticregression__C': ss.expon(scale=100)},
                n_iter = 10,
                cv = 5,
                n_jobs = -1),
    
        # third model - MLP classifier with standard scaler
        RandomizedSearchCV(
                make_pipeline(StandardScaler(),MLPClassifier()),
                param_distributions = {'mlpclassifier__alpha': ss.expon(scale=100),
                                       'mlpclassifier__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                                       'mlpclassifier__activation': ['tanh', 'relu'],
                                       'mlpclassifier__solver': ['sgd', 'adam'],
                                       'mlpclassifier__learning_rate': ['constant','adaptive'],},
                n_iter = 20,
                cv = 5,
                n_jobs = -1),
                
        # fourth model - MLP classifier                           
        RandomizedSearchCV(
                MLPClassifier(),
                param_distributions = {'alpha': ss.expon(scale=100),
                                       'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                                       'activation': ['tanh', 'relu'],
                                       'solver': ['sgd', 'adam'],
                                       'learning_rate': ['constant','adaptive'],},
                n_iter = 10,
                cv = 5,
                n_jobs = -1)
        ]


# -------------------- generata synthetic data ---------------------- # 

# produce synthetic data 
data = synthetic_data_generator(size = size,dgp = dgp,parameters = true_parameters,plots = True) 

# -------------------- preprocessing data --------------------------- # 

# get the splin between train and test data
X_train, X_test, y_train, y_test = train_test_split(data[:,1:],data[:,0],test_size = test_size)
log.info("X train shape %s",X_train.shape) ; log.info("X test shape %s",X_test.shape)
log.info("y train shape %s",y_train.shape) ; log.info("y test shape %s",y_test.shape)


# ------------------- evaluate models ----------------------------- # 

for model in models:
    # evaluate model and plot results
    model_test_validate(model,X_train,y_train,X_test,y_test)

    
    