#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:45:23 2017

@author: xujinhui
"""

import os
os.getcwd()
os.chdir('/Users/xujinhui/eecs289/hw01-data/')

import scipy.io
import numpy as np

########Q6.1
##get data 
mdict = scipy.io.loadmat("system_identification_programming_a.mat")
x=mdict['x'][0]
u=mdict['u'][0]

##set data for AX=b, where x_t1 is b, A=c(x_t, u_t), X=c(beta1,beta2)
x_t1=x[1:]; x_t=x[:-1]; u_t=u[:-1]

A_solve= np.vstack((x_t, u_t)).T
x_solve = np.linalg.inv(A_solve.T.dot(A_solve)).dot(A_solve.T).dot(x_t1)
A,B=x_solve
print('A:{},B:{}'.format(A,B))


########Q6.2
##get data
mdict_2 = scipy.io.loadmat('system_identification_programming_b.mat')
X_raw=mdict_2['x']
U_raw=mdict_2['u']

##set basic data
X_raw= X_raw.reshape(X_raw.shape[:-1])
U_raw= U_raw.reshape(U_raw.shape[:-1])

X_curr=X_raw[:-1]
U_curr=U_raw[:-1]
Y=X_raw[1:]

X=np.hstack((X_curr,U_curr))

soln = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y).T
A=soln[:,:3]; B=soln[:, 3:]
print('A: \n{}, \nB: \n{}'.format(A, B))
