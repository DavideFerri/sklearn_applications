#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:24:35 2020

@author: davideferri
"""

import numpy as np 
import pandas as pd
from SynthetData_Gen import synthetic_data_generator
#import matplotlib.pyplot as plt

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------------- configuration ------------------------------ # 

# set the sample size
size = 1000
# set the dgp
dgp = "quadratic"
# set the parameters
true_parameters = {"alpha" : 2, "beta1" : 2, "beta2" : 4} 

# -------------------- generata synthetic data ---------------------- # 

# produce synthetic data 
data = synthetic_data_generator(size = size,dgp = dgp,parameters = true_parameters,plots = True) 
