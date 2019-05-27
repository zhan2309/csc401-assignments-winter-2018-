from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import math
from scipy.special import logsumexp
import pickle

#dataDir = '/u/cs401/A3/data/'
dataDir = '../data/'
class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

def precomputeOne(theta, m):
    num_d = theta.Sigma.shape[1]
    term_2 = num_d / 2.0 * math.log(2 * math.pi)
    term_1 = 0.0
    for n in range(num_d):
        term_1 += (theta.mu[m, n] ** 2.0) / (2.0 * theta.Sigma[m, n])
    term_3 = 0.5 * math.log(np.multiply.reduce(theta.Sigma[m]))
    return term_1 + term_2 + term_3


def precomputeM(theta):
    pre_computed = []
    num_m = theta.Sigma.shape[0]
    for m in range(num_m):
        pre_computed.append(precomputeOne(theta, m))

    return pre_computed

def log_b_m_x_for_all(M,X, myTheta, precompute):
    log_Bs = np.zeros((M, X.shape[0]))
    for m in range(M):

        term_1 = 0.5 * np.matmul(1.0/myTheta.Sigma[m], np.square(X).transpose())

        term_2 = np.matmul((1.0/ myTheta.Sigma[m] * myTheta.mu[m]), X.transpose())

        log_bs = - (term_1 - term_2) - precompute[m]
        log_Bs[m] = log_bs
    return log_Bs



def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout
        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    #print ( 'TODO' )
    # pre = 0.0
    # if len(preComputedForM) > 0:
    #     pre = preComputedForM[m]
    # else:
    #     pre = precomputeOne(myTheta,m)

    # term_1 = 0.5 * np.matmul(1.0 / myTheta.Sigma[m], np.square(x).transpose())
    # term_2 = np.matmul((1.0 / myTheta.Sigma[m] * myTheta.mu[m]), x.transpose())

    # result = - (term_1 - term_2) - pre
    # return result
    add = lambda x,y,z: x+y+z
    pre = 0
    if preComputedForM == []:
        num_d = theta.Sigma.shape[1]
        term_2 = num_d / 2.0 * math.log(2 * math.pi)
        term_1 = 0.0
        for n in range(num_d):
            term_1 += (theta.mu[m, n] ** 2.0) / (2.0 * theta.Sigma[m, n])
        term_3 = 0.5 * math.log(np.multiply.reduce(theta.Sigma[m]))
        pre = add(term_1,term_2,term_3)
    else:
        pre = preComputedForM[m]    

    term_1 = 0.5 * np.matmul(1.0 / myTheta.Sigma[m], np.square(x).transpose())
    term_2 = np.matmul((1.0 / myTheta.Sigma[m] * myTheta.mu[m]), x.transpose())
    term = term_1 - term_2
    result = - term - pre
    return result


def log_p_m_x_precompute(myTheta, log_Bs):
    upper = np.add(np.log(myTheta.omega), log_Bs)
    log_Ps = upper - logsumexp(upper, axis=0)
    return log_Ps

    
def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    #print ( 'TODO' )
    pre_com = precomputeM(myTheta)

    upper = np.log(myTheta.omega[m]) + log_b_m_x(m,x,myTheta, pre_com)
    print("upper%f %f %f"%(upper,np.log(myTheta.omega[m]), log_b_m_x(m,x,myTheta, pre_com)))

    log_bs = np.zeros(myTheta.Sigma.shape[0])

    for m in range(myTheta.Sigma.shape[0]):
        log_bs[m] = log_b_m_x(m,x,myTheta, pre_com)
    lower = np.log(np.sum(np.exp((np.log(myTheta.omega.transpose()) + log_bs))))

    return upper - lower

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x
        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).
        We don't actually pass X directly to the function because we instead pass:
        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 
        See equation 3 of the handout
    '''
    p = logsumexp(log_Bs + np.log(myTheta.omega), axis=0)
    return np.sum(p)


def train( speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta( speaker, M, X.shape[1] )
    #print ('TODO')
    i=0
    prev_L = float("-inf")
    improv = float("inf")

    #init
    myTheta.omega = np.full((M, 1), 1/M)
    myTheta.Sigma = np.ones((M,d))
    rand_ind = np.random.choice(X.shape[0], M, replace=False)
    myTheta.mu = X[rand_ind]


    while i < maxIter and improv >= epsilon:
        print("===========  iteration =============%d"%i)
        #log_Bs = np.zeros((M, X.shape[0]))

        # compute intermediate
        pre_compute = precomputeM(myTheta)

        log_Bs = log_b_m_x_for_all(M,X,myTheta, pre_compute)

        log_Ps = log_p_m_x_precompute(myTheta, log_Bs)

        # compute L
        L = logLik(log_Bs, myTheta)
        #print(L)
        # update param
        log_Ps_exp = np.exp(log_Ps)

        #calculate w
        for m in range(M):
            myTheta.omega[m] = np.sum(log_Ps_exp[m])/X.shape[0]

        for m in range(M):
            upper_vec = np.dot(log_Ps_exp[m], X)
            myTheta.mu[m] = np.divide(upper_vec, np.sum(log_Ps_exp[m]))

        #calculate sigma
        for m in range(M):
            upper_vec = np.dot(log_Ps_exp[m], np.square(X))
            myTheta.Sigma[m] = np.subtract(np.divide(upper_vec, np.sum(log_Ps_exp[m])), np.square(myTheta.mu[m]))
        improv = L - prev_L
        prev_L = L
        i += 1

    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 
        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    #print ('TODO')
    liks = []

    #compute loglik
    for model in models:
        pre_compute = precomputeM(model)
        log_Bs = log_b_m_x_for_all(model.Sigma.shape[0], mfcc, model, pre_compute)
        liks.append(logLik(log_Bs, model))

    #find best
    bestModel = liks.index(max(liks))
    best_n = sorted(liks, reverse=True)[:k]

    fout = open("gmmLiks.txt", 'a')
    fout.write("%s\n"%models[correctID].name)
    for i in range(k):
        fout.write("%s %f\n"%(models[liks.index(best_n[i])].name, best_n[i]))
    fout.close()

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []

    d = 13
    k = 4  # number of top speakers to display, <= 0 if none
    M = 7
    epsilon = 0
    maxIter = 16
    count = 0

    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            if count >= 32:
                break
            print(speaker)
            count += 1

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    out = open("gmmLiks.txt", "w")
    out.close()

    # evaluate 
    numCorrect = 0;
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print (numCorrect)
    print("====================================")
    print(len(testMFCCs))
    print("accuracy %f \n" % accuracy)

