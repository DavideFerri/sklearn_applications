#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:11:33 2020

@author: davideferri
"""

import numpy as np 
import pandas as pd
import scipy.stats as ss 
import logging
import math
import matplotlib.pyplot as plt

# --------------------------- utilities ------------------------------------- #

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inv_sigmoid(y):
    x = np.log(y/(1-y))
    return x

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

def synthetic_data_generator(size,dgp,parameters,noise,plots = True):
    """
    synthetic_data_generator(size,dgp,plots = True)
    
    # ------- PARAMETERS -------- #
    
    size: int
        size of the sample
    
    dgp: str
        data generating process ; "sigmoid", "quadratic", "noise"
        
    parameters: dict
        true parameters, dict keys are "alpha","beta1","beta2"
        
    plots : Bool 
        True if you wants the plots to be savedù
        
    noise : float
        Noise coefficient
    
    # ------- OUTPUT ------------ # 
    
    data: np.array 
        data generated   
    """
    # get parameters 
    alpha = parameters["alpha"] ; beta1 = parameters["beta1"] ; beta2 = parameters["beta2"]
    
    # --------------------------- generate data on independent variables --------------------------- #
    
    # generate the independent variables
    x1 = np.random.normal(loc = 200, scale = 100, size = size)
    x2 = np.linspace(-300,300,size)
    
    # generate the y from the data based on the case
    if dgp == "sigmoid":
        # case 1 use a simple logistic
        true_values = sigmoid(alpha + beta1 * x1 + beta2 * x2)
        values = sigmoid(alpha + beta1 * x1 + beta2 * x2 + np.random.normal(loc = 0, scale = noise * np.abs((x1 - x1.mean()))))
        # set threshold 
        threshold = values.mean()
        y = (values >= threshold).astype(int)
        
    elif dgp == "quadratic": 
        # case 2 circle
        true_values = alpha + beta1 * x1**2 + beta2 * x2**2
        values = true_values + np.random.normal(loc = 0, scale = noise * (x1 - x1.mean())**2)
        # set threshold 
        threshold = values.mean()
        y = (values >= threshold).astype(int)
        
    elif dgp == "noise":
        # purely random y
        y = ss.bernoulli.rvs(p = 0.5, size = size)
        
    else:
        raise Exception
        
    # --------------------------- plot the data ---------------------------------------- # 
    
    if plots == True:
        _,ax = plt.subplots()
        ax.scatter(x1[np.where(y == 0)],x2[np.where(y == 0)],color="blue")
        ax.scatter(x1[np.where(y == 1)],x2[np.where(y == 1)],color="yellow")
        if (dgp == "sigmoid"): 
            ax.plot(x1[np.argsort(x1)],(inv_sigmoid(threshold) - alpha - beta1 * x1[np.argsort(x1)])/beta2)
            boundary = []
            boundary.append(np.array([x1[np.argsort(x1)],(inv_sigmoid(threshold) - alpha - beta1 * x1[np.argsort(x1)])/beta2]).T)
        elif (dgp == "quadratic"):
            ax.plot(x1[np.argsort(x1)],np.sqrt((threshold - alpha - beta1 * x1[np.argsort(x1)]**2)/beta2))
            ax.plot(x1[np.argsort(x1)],-np.sqrt((threshold - alpha - beta1 * x1[np.argsort(x1)]**2)/beta2))
            boundary = []
            boundary.append(np.array([x1[np.argsort(x1)],-np.sqrt((threshold - alpha - beta1 * x1[np.argsort(x1)]**2)/beta2)]).T)
            boundary.append(np.array([x1[np.argsort(x1)],np.sqrt((threshold - alpha - beta1 * x1[np.argsort(x1)]**2)/beta2)]).T)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        graph_name = "SyntheticData_scatter_" + dgp
        plt.savefig("/Users/davideferri/Documents/Repos/sklearn_applications/BivariateClassification_MLexample/graphs/" + graph_name)
    
    # -------------------------- return the data ------------------------------------ # 
    
    # get the data in a single array
    data = np.array([y,x1,x2]).T
    return data,boundary
    