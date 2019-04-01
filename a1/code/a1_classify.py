from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import argparse
import sys
import re
import os
import csv
import warnings

# run python3 a1_classify.py -i out.json.npz
warnings.filterwarnings("ignore")
def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    # – Accuracy: sum of diagonal / sum of all entries
    confusionMatrix = C
    total_entry_sum = np.sum(confusionMatrix)
    total_diag_sum = np.sum(np.diag(confusionMatrix))
    if total_entry_sum == 0:
        return 0.0
    else:
        return total_diag_sum / total_entry_sum

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    # – Recall(i): ith diagonal element / sum of row i
    confusionMatrix = C
    total_entry_row = confusionMatrix.sum(axis = 1)
    total_diag = np.diag(confusionMatrix)
    recall_result = total_diag / total_entry_row
    return recall_result

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    # Precision(i): ith diagonal element / sum of column i
    confusionMatrix = C
    total_entry_col = confusionMatrix.sum(axis = 0)
    total_diag = np.diag(confusionMatrix)
    precision_result = total_diag / total_entry_col
    return precision_result
    

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    #set up
    iBest = 0
    total_accuracyList = []
    data = np.load(filename)["arr_0"]
    X = data[:, 0:173]
    y = data[:, 173]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 50 )

    # 1. SVC: support vector machine with a linear kernel.
    svc_liner_dic = {}
    svc_linear = LinearSVC(max_iter=1000)
    svc_linear.fit(X_train, y_train)
    svc_linear_predict = svc_linear.predict(X_test)
    svc_confusion_matrix = confusion_matrix(y_test, svc_linear_predict)
    svc_liner_dic["accuracy"] = accuracy(svc_confusion_matrix)
    svc_liner_dic["recall"] = recall(svc_confusion_matrix)
    svc_liner_dic["precision"] = precision(svc_confusion_matrix)
    total_accuracyList.append(svc_liner_dic["accuracy"])
    # print(total_accuracyList)
    # 2. SVC: support vector machine with a radial basis function (γ = 2) kernel.
    svc_radial_dic = {}
    svc_radial = SVC(kernel="rbf", gamma=2, max_iter=1000)
    svc_radial.fit(X_train, y_train)
    svc_radial_predict = svc_radial.predict(X_test)
    svc_confusion_matrix_2 = confusion_matrix(y_test, svc_radial_predict)
    svc_radial_dic["accuracy"] = accuracy(svc_confusion_matrix_2)
    svc_radial_dic["recall"] = recall(svc_confusion_matrix_2)
    svc_radial_dic["precision"] = precision(svc_confusion_matrix_2)
    total_accuracyList.append(svc_radial_dic["accuracy"])
    # 3. RandomForestClassifier: with a maximum depth of 5, and 10 estimators.
    randomForestclass_dic = {}
    randomForestclass = RandomForestClassifier(n_estimators=10, max_depth=5)
    randomForestclass.fit(X_train, y_train)
    randomForestclass_prediction = randomForestclass.predict(X_test)
    rfc_confusion_matrix_3 = confusion_matrix(y_test, randomForestclass_prediction)
    randomForestclass_dic["accuracy"] = accuracy(rfc_confusion_matrix_3)
    randomForestclass_dic["recall"] = recall(rfc_confusion_matrix_3)
    randomForestclass_dic["precision"] = precision(rfc_confusion_matrix_3)
    total_accuracyList.append(randomForestclass_dic["accuracy"])
    # 4. MLPClassifier: A feed-forward neural network, with α = 0.05.
    mlp_class_dic = {}
    mlp_class = MLPClassifier(alpha=0.05)
    mlp_class.fit(X_train, y_train)
    mlp_class_prediction = mlp_class.predict(X_test)
    mlp_confusion_matrix_4 = confusion_matrix(y_test, mlp_class_prediction)
    mlp_class_dic["accuracy"] = accuracy(mlp_confusion_matrix_4)
    mlp_class_dic["recall"] = recall(mlp_confusion_matrix_4)
    mlp_class_dic["precision"] = precision(mlp_confusion_matrix_4)
    total_accuracyList.append(mlp_class_dic["accuracy"])
    # 5. AdaBoostClassifier: with the default hyper-parameters.
    adaBoostClass_dic = {}
    adaBoostClass = AdaBoostClassifier()
    adaBoostClass.fit(X_train, y_train)
    adaBoostClass_prediction = adaBoostClass.predict(X_test)
    adaBoostClass_confusion_matrix_5 = confusion_matrix(y_test, adaBoostClass_prediction)
    adaBoostClass_dic["accuracy"] = accuracy(adaBoostClass_confusion_matrix_5)
    adaBoostClass_dic["recall"] = recall(adaBoostClass_confusion_matrix_5)
    adaBoostClass_dic["precision"] = precision(adaBoostClass_confusion_matrix_5)
    total_accuracyList.append(adaBoostClass_dic["accuracy"])

    print(total_accuracyList)
    # get best one

    iBest = total_accuracyList.index(max(total_accuracyList)) + 1
    # write csv file
    with open("a1_3.1.csv", 'w') as f:
        fw = csv.writer(f, delimiter=',')
        fw.writerow([1, svc_liner_dic["accuracy"]] + list(svc_liner_dic["recall"]) + list(svc_liner_dic["precision"]) + list(svc_confusion_matrix[0]) + list(svc_confusion_matrix[1]) + list(svc_confusion_matrix[2]) + list(svc_confusion_matrix[3]))
        fw.writerow([2, svc_radial_dic["accuracy"]] + list(svc_radial_dic["recall"]) + list(svc_radial_dic["precision"]) + list(svc_confusion_matrix_2[0]) + list(svc_confusion_matrix_2[1]) + list(svc_confusion_matrix_2[2]) + list(svc_confusion_matrix_2[3]))
        fw.writerow([3, randomForestclass_dic["accuracy"]] + list(randomForestclass_dic["recall"]) + list(randomForestclass_dic["precision"]) + list(rfc_confusion_matrix_3[0]) + list(rfc_confusion_matrix_3[1]) + list(rfc_confusion_matrix_3[2]) + list(rfc_confusion_matrix_3[3]))
        fw.writerow([4, mlp_class_dic["accuracy"]] + list(mlp_class_dic["recall"]) + list(mlp_class_dic["precision"]) + list(mlp_confusion_matrix_4[0]) + list(mlp_confusion_matrix_4[1]) + list(mlp_confusion_matrix_4[2]) + list(mlp_confusion_matrix_4[3]))
        fw.writerow([5, adaBoostClass_dic["accuracy"]] + list(adaBoostClass_dic["recall"]) + list(adaBoostClass_dic["precision"]) + list(adaBoostClass_confusion_matrix_5[0]) + list(adaBoostClass_confusion_matrix_5[1]) + list(adaBoostClass_confusion_matrix_5[2]) + list(adaBoostClass_confusion_matrix_5[3]))
    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''
    svc_linear = LinearSVC(max_iter=1000)
    svc_radial = SVC(kernel="rbf", gamma=2, max_iter=1000)
    randomForestclass = RandomForestClassifier(n_estimators=10, max_depth=5)
    mlp_class = MLPClassifier(alpha=0.05)
    adaBoostClass = AdaBoostClassifier()
    classifierList = [svc_linear, svc_radial, randomForestclass, mlp_class, adaBoostClass]
    data_num_list = [1000, 5000, 10000, 15000,20000]
    index_classifier = iBest - 1
    total_accuracyList = []
    for data_set in data_num_list:
        index = np.random.choice(list(range(len(X_train))), size=data_set)
        X_sampled = X_train[index]
        y_sampled = y_train[index]
        classifierList[index_classifier].fit(X_sampled, y_sampled)
        classifier_prediction = classifierList[index_classifier].predict(X_test)
        confusionMatrix = confusion_matrix(y_test, classifier_prediction)
        total_accuracyList.append(accuracy(confusionMatrix))
        if data_set == 1000:
            X_1k = X_sampled
            y_1k = y_sampled
    # write csv file
    with open("a1_3.2.csv", 'w') as f:
        fw = csv.writer(f, delimiter=',')
        fw.writerow(list(total_accuracyList))
        temp = "when the data set goes up the accuracy is increasing however, the strength of\
        increasing is decreased. The reson might be that the centre limit theorem as the sample large enough\
        , the statics are tending for the real population"
        temp_ = re.sub(' +', ' ', temp)
        fw.writerow(["explanation: ",temp_])

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    svc_linear = LinearSVC(max_iter=1000)
    svc_radial = SVC(kernel="rbf", gamma=2, max_iter=1000)
    randomForestclass = RandomForestClassifier(n_estimators=10, max_depth=5)
    mlp_class = MLPClassifier(alpha=0.05)
    adaBoostClass = AdaBoostClassifier()
    classifierList = [svc_linear, svc_radial, randomForestclass, mlp_class, adaBoostClass]

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    svc_linear = LinearSVC(max_iter=1000)
    svc_radial = SVC(kernel="rbf", gamma=2, max_iter=1000)
    randomForestclass = RandomForestClassifier(n_estimators=10, max_depth=5)
    mlp_class = MLPClassifier(alpha=0.05)
    adaBoostClass = AdaBoostClassifier()
    classifierList = [svc_linear, svc_radial, randomForestclass, mlp_class, adaBoostClass]
    iBest = i - 1
    data = np.load(filename)["arr_0"]
    X = data[:,:173]
    y = data[:,173]
    total_accs = []
    k_fold = KFold(n_splits=5, shuffle=True)
    for train_i, test_i in k_fold.split(X):
        X_train = X[train_i]
        X_test = X[test_i]
        y_train = y[train_i]
        y_test = y[test_i]
        accs = []
        for classifier in classifierList:
            classifier.fit(X_train, y_train)
            classifier_prediction = classifier.predict(X_test)
            c_confusionMatrix = confusion_matrix(y_test, classifier_prediction)
            classifier_accuracy = accuracy(c_confusionMatrix)
            accs.append(classifier_accuracy)
        total_accs.append(accs)
    with open("a1_3.4.csv", 'w') as f:
        fw = csv.writer(f, delimiter=',')
        for accs in total_accs:
            fw.writerow(accs)
        comparing_list = []
        k = 0
        while k != iBest and k < 5:
            # S = stats.ttest_rel(a, b)
            S = stats.ttest_rel(total_accs[k], total_accs[iBest])
            comparing_list.append(S)
            k += 1
        fw.writerow(comparing_list)
        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    class34(args.input, iBest)