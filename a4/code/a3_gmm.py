from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import math
from scipy.special import logsumexp
import pickle

dataDir = '/u/cs401/A3/data/'
# dataDir = '../data/'
class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))



def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout
        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    #print ( 'TODO' )
    add = lambda x,y,z: x+y+z
    previous_com = 0
    const_ = 2
    if preComputedForM == []:
        shape_ = theta.Sigma.shape[1]
        b = shape_ / const_ * math.log(const_ * math.pi)
        a = 0
        i = 0
        while i < (shape_):
            power = lambda x,y: x**y
            x_1 = (power(theta.mu[m, i],const_))
            y = (const_ * theta.Sigma[m, i])
            if y == 0:
                break
            a +=  (x_1 / y)
            i+=1
        c = (1/2) * math.log(np.multiply.reduce(theta.Sigma[m]))
        previous_com = add(a,b,c)
    else:
        previous_com = preComputedForM[m]    
    temp = 1 / myTheta.Sigma[m]
    a = (1/2) * np.matmul(temp, np.square(x).transpose())
    b = np.matmul((temp * myTheta.mu[m]), x.transpose())
    term = a - b
    result = - term - previous_com
    return result

    
def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    # [-1.3267119]
    # [-0.5837369]
    
    omegaT = myTheta.omega.transpose()
    log_b_s = []
    shape_dic = {"shape_1": myTheta.Sigma.shape[0], "shape_2":myTheta.Sigma.shape[1]}
    preComputation = []
    divi = lambda a,b: np.divide(a, b)

    const_ = 2
    k = 0
    while k < (shape_dic["shape_1"]):
        square_ = lambda a,b: np.power(a, 2)
        temp = const_ * myTheta.Sigma[k]
        term_1 = np.sum(divi(square_(myTheta.mu[k], const_), temp))
        temp_0 = np.log(const_ * np.pi) * shape_dic["shape_2"] / const_
        f_temp = term_1 + temp_0
        result = f_temp + (1/2) * np.log(np.prod(myTheta.Sigma[k]))
        preComputation += [result]
        k +=1

    i = 0
    while i < (shape_dic["shape_1"]):
        log_b = log_b_m_x(i, x, myTheta, preComputation)
        log_b_s += [log_b]
        i += 1

    log_b_omega = log_b_s + np.log(omegaT)
    x = logsumexp(log_b_omega)
    log_b_theta = np.log(myTheta.omega[m]) + log_b_s[m]
    logP = log_b_theta - x
    # print(logP)
    return logP
    

    



    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x
        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).
        We don't actually pass X directly to the function because we instead pass:
        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 
        See equation 3 of the handout
    '''
    log_omega = np.log(myTheta.omega)
    logPs = logsumexp(log_omega + log_Bs, axis=0)
    return np.sum(logPs)


def train( speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    # tut formula
    shape0,shape1 = X.shape
    myTheta = theta(speaker, M, shape1)
    #print ('TODO')
    iteration=0
    previousL = np.NINF
    improvement = np.Infinity

    #initialize theta
    myTheta.omega = np.full((M, 1), 1/M)
    myTheta.Sigma[:, :] = 1.
    myTheta.mu = X[np.random.choice(X.shape[0], M, replace=False)]
    shape_dic = {"shape_1": myTheta.Sigma.shape[0], "shape_2":myTheta.Sigma.shape[1]}
    const_ = 2
    log_p_theta = np.log(myTheta.omega)
    while (iteration < maxIter) and (improvement >= epsilon):
        log_Bs = np.zeros((M, X.shape[0]))

        pre_computed = []
        add = lambda x,y,z: x+y+z
        power = lambda x,y: x**y
        
        each_m = 0
        while each_m < (shape_dic["shape_1"]):
            if (const_ * math.log(const_ * math.pi)) == 0:
                break
            b_ = shape_dic["shape_1"] / const_ * math.log(const_ * math.pi)
            a_ = 0
            i = 0
            while i < (shape_dic["shape_2"]):
                a_ += (power(myTheta.mu[each_m, i], const_)) / (const_ * myTheta.Sigma[each_m, i])
                if ((const_ * myTheta.Sigma[each_m, i])) == 0:
                    break
                i+=1

            threeTerms = add(a_,b_,((1/const_) * math.log(np.multiply.reduce(myTheta.Sigma[each_m]))))
            pre_computed.append(threeTerms)
            each_m += 1

        # get all log_Bs
        b_m = 0
        while b_m < M:
            temp = 1 / myTheta.Sigma[b_m]
            a = (1/const_) * np.matmul(temp, np.square(X).transpose())
            b = np.matmul((temp * myTheta.mu[b_m]), X.transpose())
            term = a - b
            result = - term - pre_computed[b_m]
            log_Bs[b_m] = result
            b_m += 1

    
        # get log_Ps
        log_Ps = log_p_theta + log_Bs - logsumexp((log_p_theta + log_Bs), axis=0)

        # compute for L and change improvement
        logPs = logsumexp(log_p_theta + log_Bs, axis=0)
        L = np.sum(logPs)
        improvement = L - previousL
        previousL = L
        m = 0

        while m < (M):
            # omega
            shape_0, _ = X.shape
            myTheta.omega[m] = np.sum(np.exp(log_Ps)[m])/shape_0
            # mu
            dotProduct = np.dot(np.exp(log_Ps)[m], X)
            myTheta.mu[m] = np.divide(dotProduct, np.sum(np.exp(log_Ps)[m]))
            # Sigma
            term_a = np.divide(np.dot(np.exp(log_Ps)[m], power(X,const_)), np.sum(np.exp(log_Ps)[m]))
            term_b = power(myTheta.mu[m], const_)
            myTheta.Sigma[m] = (term_a - term_b)
            m += 1
        iteration += 1

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
    # want to find bestModel
    flag = 0
    bestModelIndex = 0
    l_s = []
    const_ = 2
    for model in models:
        b_m = 0
        myTheta = model
        mfcc_shape, _ = mfcc.shape
        shape_dic = {"shape_1": myTheta.Sigma.shape[0], "shape_2":myTheta.Sigma.shape[1]}
        log_Bs = np.zeros((shape_dic["shape_1"], mfcc_shape))
       
        pre_computed = []
        add = lambda x,y,z: x+y+z
        power = lambda x,y: x**y
        
        each_m = 0
        while each_m < (shape_dic["shape_1"]):
            
            b_ = shape_dic["shape_1"] / const_ * math.log(const_ * math.pi)
            a_ = 0
            i = 0
            while i < (shape_dic["shape_2"]):
                a_ += (power(myTheta.mu[each_m, i], const_)) / (const_ * myTheta.Sigma[each_m, i])
                i+=1
            reduceM = np.multiply.reduce(myTheta.Sigma[each_m])
            threeTerms = add(a_,b_,((1/const_) * math.log(reduceM)))
            pre_computed.append(threeTerms)
            each_m += 1

        
        while b_m < shape_dic["shape_1"]:
            temp = 1 / myTheta.Sigma[b_m]
            a = (1/const_) * np.matmul(temp, np.square(mfcc).transpose())
            b = np.matmul((temp * myTheta.mu[b_m]), mfcc.transpose())
            term = a - b
            result = - term - pre_computed[b_m]
            log_Bs[b_m] = result
            b_m += 1
        

        logPs = logsumexp(np.log(myTheta.omega) + log_Bs, axis=0)
        L = np.sum(logPs)
        l_s.append(L)
    
    for i in range(len(l_s)):
        if l_s[i] > l_s[bestModelIndex]:
            bestModelIndex = i
    

    # write file
    if k > 0:
        out = open("gmmLiks.txt", 'a')
        out.write("%s\n"%models[correctID].name)
        time = 0
        k_times = k
        while time < (k_times):
            out.write("%s "%(models[l_s.index(sorted(l_s, reverse=True)[:k][time])].name))
            out.write("%f\n"%(sorted(l_s, reverse=True)[:k][time]))
            time +=1
        out.close()
    # check
    if bestModelIndex == correctID:
        flag = 1
    else:
        return 0
    return flag


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []

    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 3
    epsilon = 0
    maxIter = 1

    number_of_speaker = 0
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)
            # if number_of_speaker > 8:
            #     break
            # number_of_speaker +=1

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
    print("================Result==============")
    print(len(testMFCCs))
    print(accuracy)

