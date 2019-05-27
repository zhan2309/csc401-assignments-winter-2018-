from a3_gmm import *
from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import math
from scipy.special import logsumexp

if __name__ == "__main__":

    M = 3
    d = 2
    myTheta = theta("a", M, d)
    myTheta.omega = np.array([[1],[2], [3]], dtype='f')
    myTheta.mu = np.array([[1,2], [3,4], [6,7]], dtype='f')
    myTheta.Sigma = np.array([[5,6], [7, 8], [8,9]], dtype='f')

    X = np.array([[2,4], [3,5]], dtype='f')

    log_Bs = np.zeros((M, X.shape[0]))
    pre_compute = precomputeM(myTheta)
    print(pre_compute)



    log_Bs = log_b_m_x_for_all(M, X, myTheta, pre_compute)
    print(log_Bs)

    log_B2 = np.zeros((M,X.shape[0]))
    for x_ind, x in enumerate(X):
        for m in range(M):
            log_B2[m, x_ind] = log_b_m_x(m, x, myTheta, pre_compute)
    print("---------------------")
    print(log_B2)
    print("----------------------")




    log_p2 = np.zeros((M, X.shape[0]))
    for x_ind, x in enumerate(X):
        for m in range(M):
            log_p2[m, x_ind] = log_p_m_x(m, x, myTheta)

    print("test p no use")
    print(log_p2)


    print("test p")
    log_p = log_p_m_x_precompute(myTheta, log_Bs)
    print(log_p)