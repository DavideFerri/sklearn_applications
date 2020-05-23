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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn
#import matplotlib.pyplot as plt

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,format='%(name)s - %(levelname)s - %(message)s')

print(sorted(sklearn.metrics.SCORERS.keys()))
# --------------------- utilities --------------------------------- # 


def model_test_validate(model,X_train,y_train,X_test,y_test,true_boundary):
    """
    
    """
    #log.debug("The available parameters for cv are: %s", model.get_params().keys())
    # fit model
    model.fit(X_train,y_train)
    # get performance info on the cross-validation exercise
    log.info("The best parameters chosen in cross-validation are: %s", model.best_params_)
    log.info("The performance of the model in cross-validation is: %s", model.best_score_)
    # make predictions
    y_test_pred = model.predict(X_test)
    # get the performance on the test dataset
    log.info("The performance of the model in test is: %s", model.score(X_test,y_test))
    # plt test labels vs predicted labels
    _,ax = plt.subplots(1,2)
    for y,i,name in zip([y_test,y_test_pred],np.arange(2),["test labels","predicted labels"]):
        ax[i].plot(   X_test[np.where(y == 0),0],
                      X_test[np.where(y == 0),1],
                      ".",color = "blue")
        ax[i].plot(   X_test[np.where(y == 1),0],
                      X_test[np.where(y == 1),1],
                      ".",color = "yellow")
        # plot true decision boundary 
        for elem in boundary:
            ax[i].plot(elem[:,0],elem[:,1])
    # plt correct vs wrong labels
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
dgp = "sigmoid"
# set the parameters
true_parameters = {"alpha" : 2, "beta1" : 2, "beta2" : 4} 
# set the noise coeff
noise = 20
# set the test size as a proportion 
test_size = 0.2

# ------------------- specify estimators/models ----------------------------- # 

models = [
        # first model - Linear SVC with standard scaler
        RandomizedSearchCV(
                make_pipeline(StandardScaler(),LinearSVC(max_iter = 500)),
                param_distributions = {'linearsvc__C': ss.expon(scale=100)},
                scoring = ['f1','accuracy'],
                n_iter = 10,
                refit = 'f1',
                cv = 5,
                n_jobs = -1),            
        
        #  second model - Logistic regression with standard scaler
        RandomizedSearchCV(
                make_pipeline(StandardScaler(),LogisticRegression(solver = 'lbfgs',max_iter = 500)),
                param_distributions = {'logisticregression__C': ss.expon(scale=100)},
                scoring = ['f1','accuracy'],
                n_iter = 10,
                refit = 'f1',
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
                scoring = ['f1','accuracy'],
                n_iter = 20,
                refit = 'f1',
                cv = 5,
                n_jobs = -1),
                
        # fourth model - MLP classifier                           
        RandomizedSearchCV(
                MLPClassifier(),
                param_distributions = {'alpha': ss.expon(scale=100),
                                       'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                                       'activation': ['tanh', 'relu'],
                                       'solver': ['sgd', 'adam'],
                                       'learning_rate': ['constant','adaptive']},
                scoring = ['f1','accuracy'],           
                n_iter = 10,
                refit = 'f1',
                cv = 5,
                n_jobs = -1),
        """
        # fifth model - Decision tree Classifier with standard scaler
        
        GridSearchCV(
                make_pipeline(StandardScaler(),DecisionTreeClassifier()),
                param_grid = {'decisiontreeclassifier__min_samples_split': np.arange(2, 10), 'decisiontreeclassifier__min_samples_leaf': 
                                        np.arange(.05, .2), 'decisiontreeclassifier__max_leaf_nodes': np.arange(2, 30)},
                scoring = ["f1","accuracy"],
                refit = "f1",
                cv = 5,
                n_jobs = -1),
            
        # sixth model - Random Forest with standard scaler
        
        GridSearchCV(
                make_pipeline(StandardScaler(),RandomForestClassifier()),
                param_grid = {'randomforestclassifier__n_estimators': np.arange(10, 20), 'randomforestclassifier__min_samples_split': np.arange(2, 10),
                              'randomforestclassifier__min_samples_leaf': np.arange(.15, .33), 'randomforestclassifier__max_leaf_nodes': np.arange(2, 30),
                              'randomforestclassifier__bootstrap': ['True', 'False']},
                scoring = ["f1","accuracy"],
                refit = "f1",
                cv = 5,
                n_jobs = -1),
        
        # seventh model - Bagging Classifier with standard scaler
        
        GridSearchCV(
                make_pipeline(StandardScaler(),BaggingClassifier()),
                param_grid = {'baggingclassifier__n_estimators': np.arange(10, 30), 'baggingclassifier__max_samples': np.arange(2, 
                                       30), 'baggingclassifier__bootstrap': ['True', 'False'], 'baggingclassifier__bootstrap_features': ['True', 
                                        'False']},
                scoring = ["f1","accuracy"],
                refit = "f1",
                cv = 5,
                n_jobs = -1),
    
        # eight model - AdaBoost with standard scaler
        
        GridSearchCV(
                make_pipeline(StandardScaler(),AdaBoostClassifier()),
                param_grid = {'adaboostclassifier__n_estimators': np.arange(10, 30), 'adaboostclassifier__learning_rate': 
                                        np.arange(.05, .1)},
                scoring = ["f1","accuracy"],
                refit = "f1",
                cv = 5,
                n_jobs = -1)"""
        ]


# -------------------- generata synthetic data ---------------------- # 

# produce synthetic data 
data,boundary = synthetic_data_generator(size = size,dgp = dgp,parameters = true_parameters,noise = noise,plots = True) 
print(boundary[0].shape)

# -------------------- preprocessing data --------------------------- # 

# get the splin between train and test data
X_train, X_test, y_train, y_test = train_test_split(data[:,1:],data[:,0],test_size = test_size)
log.info("X train shape %s",X_train.shape) ; log.info("X test shape %s",X_test.shape)
log.info("y train shape %s",y_train.shape) ; log.info("y test shape %s",y_test.shape)


# ------------------- evaluate models ----------------------------- # 

for model in models:
    # evaluate model and plot results
    model_test_validate(model,X_train,y_train,X_test,y_test,boundary)

    
    