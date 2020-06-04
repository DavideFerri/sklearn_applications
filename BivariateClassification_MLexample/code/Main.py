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
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve
import logging
import matplotlib.pyplot as plt

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------------- utilities --------------------------------- # 

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def model_test_validate(model,X_train,y_train,X_test,y_test,true_boundary,graph_path):
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
    fig_name = graph_path + " test vs predicted labels"
    plt.savefig(fig_name)
    # plt correct vs wrong labels
    _,ax = plt.subplots()
    ax.plot(   X_test[np.where(y_test == y_test_pred),0],
                  X_test[np.where(y_test == y_test_pred),1],
                  ".",color = "green")
    ax.plot(   X_test[np.where(y_test != y_test_pred),0],
                  X_test[np.where(y_test != y_test_pred),1],
                  ".",color = "red")
    ax.set_xlabel("x1") ; ax.set_ylabel("x2")
    fig_name = graph_path + " correct vs wrong labels"
    plt.savefig(fig_name)
    # plt normalized confusion matrix of best estimator
    plt.figure()
    conf_matrix = plot_confusion_matrix(model.best_estimator_,X_test,y_test,cmap=plt.cm.Blues,normalize = "true")
    conf_matrix.ax_.set_title("Normalized confusion matrix of the classifier")
    fig_name = graph_path + " confusion matrix"
    plt.savefig(fig_name)
    # plot the learning curve
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plot_learning_curve(model.best_estimator_, title = "Learning curve of top estimator",
                        X = X_train, y = y_train, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=5, n_jobs=4)
    fig_name = graph_path + " learning curve"
    fig_name = graph_path + " learning curve"
    plt.savefig(fig_name)
    
    
# --------------------- configuration ------------------------------ # 

# set the sample size
size = 3000
# set the dgp
dgp = "quadratic"
# set the parameters
true_parameters = {"alpha" : 2, "beta1" : 2, "beta2" : 4} 
# set the noise coeff
noise = 10
# set the test size as a proportion 
test_size = 0.2

# ------------------- specify estimators/models ----------------------------- # 

models = {
        # first model - Linear SVC with standard scaler
        "Scaler linear SVC" :
        RandomizedSearchCV(
                make_pipeline(StandardScaler(),LinearSVC(max_iter = 500)),
                param_distributions = {'linearsvc__C': ss.expon(scale=100)},
                scoring = ['f1','accuracy'],
                n_iter = 10,
                refit = 'f1',
                cv = 5,
                n_jobs = -1),            
        
        #  second model - Logistic regression with standard scaler
        "Scaler logistic regression" :
        RandomizedSearchCV(
                make_pipeline(StandardScaler(),LogisticRegression(solver = 'lbfgs',max_iter = 500)),
                param_distributions = {'logisticregression__C': ss.expon(scale=100)},
                scoring = ['f1','accuracy'],
                n_iter = 10,
                refit = 'f1',
                cv = 5,
                n_jobs = -1),
    
        # third model - MLP classifier with standard scaler
        "Scaler MLP classifier" :
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
        "MLP classifier" :                           
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
        
        # fifth model - Decision tree Classifier with standard scaler
        #"Scaler decision tree" :
        #GridSearchCV(
         #       make_pipeline(StandardScaler(),DecisionTreeClassifier()),
         #       param_grid = {'decisiontreeclassifier__min_samples_split': np.arange(2, 10), 'decisiontreeclassifier__min_samples_leaf': 
         #                               np.arange(.05, .2), 'decisiontreeclassifier__max_leaf_nodes': np.arange(2, 30)},
         #       scoring = ["f1","accuracy"],
         #       refit = "f1",
         #       cv = 5,
         #       n_jobs = -1),
            
        # sixth model - Random Forest with standard scaler
        #"Scaler random forest" :
        #GridSearchCV(
         #       make_pipeline(StandardScaler(),RandomForestClassifier()),
         #       param_grid = {'randomforestclassifier__n_estimators': np.arange(10, 20), 'randomforestclassifier__min_samples_split': np.arange(2, 10),
         #                     'randomforestclassifier__min_samples_leaf': np.arange(.15, .33), 'randomforestclassifier__max_leaf_nodes': np.arange(2, 30),
         #                     'randomforestclassifier__bootstrap': ['True', 'False']},
         #       scoring = ["f1","accuracy"],
         #       refit = "f1",
         #       cv = 5,
         #       n_jobs = -1),
        
        # seventh model - Bagging Classifier with standard scaler
        #"Scaler bagging trees":
        #GridSearchCV(
          #      make_pipeline(StandardScaler(),BaggingClassifier(DecisionTreeClassifier())),
          #      param_grid = {'baggingclassifier__n_estimators': np.arange(10, 30), 'baggingclassifier__max_samples': np.arange(2, 
          #                             30), 'baggingclassifier__bootstrap': ['True', 'False'], 'baggingclassifier__bootstrap_features': ['True', 
          #                              'False']},
          #      scoring = ["f1","accuracy"],
          #      refit = "f1",
          #      cv = 5,
         #       n_jobs = -1),
    
        # eight model - AdaBoost with standard scaler
        #"Scaler adaboost trees":
        #GridSearchCV(
         #       make_pipeline(StandardScaler(),AdaBoostClassifier(DecisionTreeClassifier())),
         #       param_grid = {'adaboostclassifier__n_estimators': np.arange(10, 30), 'adaboostclassifier__learning_rate': 
         #                               np.arange(.05, .1)},
         #       scoring = ["f1","accuracy"],
         #       refit = "f1",
        #       cv = 5,
        #        n_jobs = -1)
        }


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

for key in models.keys():
    model = models[key]
    graph_path = "/Users/davideferri/Documents/Repos/sklearn_applications/BivariateClassification_MLexample/graphs/" + key
    if key in ["Scaler bagging trees","Scaler adaboost trees"]:
        model.set_params(estimator__base_estimator = models["Scaler decision tree"].best_estimator_)
    # evaluate model and plot results
    model_test_validate(model,X_train,y_train,X_test,y_test,boundary,graph_path)

    
    