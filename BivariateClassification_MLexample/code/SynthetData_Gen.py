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

# --------------------------- utilities ------------------------------------- #

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inv_sigmoid(y):
    x = np.log(y/(1-y))
    return x

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------------------- configuration -------------------------------------------- #

# select the size of the sample be selected 
size = 1000
# relationship type 
case = 2

# --------------------------- generate data on independent variables --------------------------- #

# generate the independent variables
x1 = np.random.normal(loc = 2, scale = 100, size = size)
x2 = np.linspace(-300,300,size)

# generate the y from the data based on the case
if case == 1:
    # get the true parameters 
    alpha = 3 ; beta1 = 4 ; beta2 = 2
    # set threshold 
    threshold = 0.5
    # case 1 use a simple logistic
    true_values = sigmoid(alpha + beta1 * x1 + beta2 * x2)
    noise = np.random.normal(loc = 0, scale = np.abs((x1 - x1.mean()))/100)
    y = (true_values + noise >= threshold).astype(int)
    
elif case == 2: 
    # get the true parameters 
    alpha = 3 ; beta1 = 1 ; beta2 = 2
    # case 2 circle
    true_values = alpha + beta1 * x1**2 + beta2 * x2**2
    noise = np.random.normal(loc = 0, scale = beta1 * 10 * (x1 - x1.mean())**2)
    # set threshold 
    threshold = (true_values + noise).mean()
    y = (true_values + noise >= threshold).astype(int)
    
elif case == 3:
    # purely random y
    y = ss.bernoulli.rvs(p = 0.5, size = size)
    
else:
    raise Exception
    
# --------------------------- plot the data ---------------------------------------- # 
  
_,ax = plt.subplots()
ax.scatter(x1[np.where(y == 0)],x2[np.where(y == 0)],color="blue")
ax.scatter(x1[np.where(y == 1)],x2[np.where(y == 1)],color="red")
if (case == 1): 
    ax.plot(x1[np.argsort(x1)],(inv_sigmoid(threshold) - alpha - beta1 * x1[np.argsort(x1)])/beta2)
elif (case == 2):
    ax.plot(x1[np.argsort(x1)],np.sqrt((threshold - alpha - beta1 * x1[np.argsort(x1)]**2)/beta2))
    ax.plot(x1[np.argsort(x1)],-np.sqrt((threshold - alpha - beta1 * x1[np.argsort(x1)]**2)/beta2))