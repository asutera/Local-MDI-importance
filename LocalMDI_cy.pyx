"""

Authors: Pierre Geurts (p.geurts@uliege.be) & Antonio Sutera (sutera.antonio@gmail.com)
License: BSD 3 clause

"""

from __future__ import print_function

import sklearn as sk
import itertools
import math
import sklearn.ensemble
import sklearn.metrics

import numpy as np

def compute_mdi_local_tree(tree, float[:,:] X, Py_ssize_t nsamples, int nfeatures, double[:,:] vimp_view):
    
    cdef Py_ssize_t i

    cdef double[:] impurity_view = tree.impurity
    cdef double[:] threshold_view = tree.threshold
    cdef long[:] children_left_view = tree.children_left
    cdef long[:] children_right_view = tree.children_right
    cdef long[:] feature_view = tree.feature

    cdef Py_ssize_t node = 0
    cdef double newvimp = 0.0
    cdef double oldvimp = 0.0
    cdef int ifeat = 0
    
    for i in range(nsamples):
        
        # We propagate the example in the tree
        node = 0
        oldvimp = impurity_view[node]
        
        while children_left_view[node] != -1:
            ifeat = feature_view[node]
            if X[i,ifeat] <= threshold_view[node]:
                node = children_left_view[node]
            else:
                node = children_right_view[node]
            newvimp = impurity_view[node]
            vimp_view[i, ifeat] += oldvimp - newvimp
            oldvimp = newvimp

def compute_mdi_local_ens(ens, X, verbose=0):

    cdef float[:,:] X_view = X
    cdef Py_ssize_t nsamples = X.shape[0]
    cdef int nfeatures = X.shape[1]
    
    vimp = np.zeros((X.shape[0],ens.n_features_),dtype='float64')
    cdef double[:,:] vimp_view = vimp
    cdef int nestimators = ens.n_estimators
    
    for i in range(nestimators):
        if verbose>0:
            print("o",end='',flush=True)
        compute_mdi_local_tree(ens.estimators_[i].tree_, X_view,
                               nsamples, nfeatures, vimp_view)

    print("")
    vimp /= ens.n_estimators
    return vimp

def compute_shapley_local_tree(tree, float[:,:] X, Py_ssize_t nsamples, int nfeatures, double[:,:] vimp_view, int clas_view):
    

    cdef Py_ssize_t i

    cdef double[:,:,:] value_view = tree.value
    cdef double[:] threshold_view = tree.threshold
    cdef long[:] children_left_view = tree.children_left
    cdef long[:] children_right_view = tree.children_right
    cdef long[:] feature_view = tree.feature

    cdef Py_ssize_t node = 0
    cdef double newvimp = 0.0
    cdef double oldvimp = 0.0
    cdef int ifeat = 0
    
    for i in range(nsamples):
        
        node = 0
        oldvimp = value_view[node][0][clas_view]
        
        while children_left_view[node] != -1:
            ifeat = feature_view[node]
            if X[i,ifeat] <= threshold_view[node]:
                node = children_left_view[node]
            else:
                node = children_right_view[node]
            newvimp = value_view[node][0][clas_view]
            # Attention we have to swap them here (wrt vimp)
            vimp_view[i, ifeat] += newvimp - oldvimp
            oldvimp = newvimp

def compute_shapley_local_ens(ens, X, clas=0):

    cdef float[:,:] X_view = X
    cdef Py_ssize_t nsamples = X.shape[0]
    cdef int nfeatures = X.shape[1]
    cdef int clas_view = clas
    
    vimp = np.zeros(X.shape,dtype='float64')
    cdef double[:,:] vimp_view = vimp
    cdef int nestimators = ens.n_estimators
    
    for i in range(nestimators):
        print(".",end='',flush=True)
        compute_shapley_local_tree(ens.estimators_[i].tree_, X_view,
                                   nsamples, nfeatures, vimp_view, clas_view)

    print("")
    vimp /= ens.n_estimators
    return vimp

