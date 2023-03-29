from warnings import simplefilter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from numpy import genfromtxt
from numpy.lib.recfunctions import append_fields
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate

simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=np.inf)

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

df = pd.read_csv('NF-BoT-IoT.csv', header=0, dtype=float, usecols=range(4, 12))
dt = pd.read_csv('NF-BoT-IoT.csv', header=0, dtype=int, usecols=[12])

plt.figure()
plt.boxplot([df['IN_PKTS'],df['OUT_PKTS']], notch=True,patch_artist=True)
plt.show()

####Scale feature matrix######
feature_std = StandardScaler().fit_transform(df.values)

labels = LabelEncoder().fit_transform(dt.values)


x_train, x_test, y_train, y_test = train_test_split(feature_std, labels, test_size=0.34, random_state=42, shuffle=True)


print("Begin:__________________________________")

def print_stats_metrics(y_test, y_pred):    
    print('Accuracy: %.2f' % accuracy_score(y_test,y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
    tp_rate = tp/(tp+fn)
    tn_rate = tn/(tn+fp) 
    g_mean = np.sqrt(tp_rate*tn_rate)
    print ("confusion matrix")
    print(confmat)
    print (pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='binary'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
    print('Geometric mean: %.3f' % g_mean)
    print('AUC score: %.3f' % roc_auc_score(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % auc(fpr, tpr))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def cross_validation(model, _X, _y, _cv=5):

    _scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = cross_validate(estimator=model,
                            X=_X,
                            y=_y,
                            cv=_cv,
                            scoring=_scoring)
    
    plt.figure()
    box_plot_data=[results['test_accuracy'],results['test_precision'],results['test_recall'],results['test_f1'],results['test_roc_auc']]
    plt.boxplot(box_plot_data,notch=True,patch_artist=True,labels=['accuracy','precision','recall','f1-score','roc-auc'])
    plt.show()

    print("Accuracy:", results['test_accuracy'])
    print("Precision:", results['test_precision'])
    print("Recall:", results['test_recall'])
    print("F1-measure:", results['test_f1'])
    print("AUC score:", results['test_roc_auc'])
      

algorithm = int(input("Choose algorithm:\n1-Naive Bayes\n2-Random Fores\n3-KNN\n4-Decision Tree\nEnter number: "))

if(algorithm == 1):
    #######################Naive Bayes#######################
    clfNB = GaussianNB()
    clfNB.fit(x_train,y_train)
    predictions = clfNB.predict(x_test)
    print("#######################Naive Bayes#######################")
    print_stats_metrics(y_test, predictions)
    crossValidation = input("Run 10-k Fold?\nY/N: ")
    if(crossValidation == 'Y'):
        predictions = cross_validation(clfNB, feature_std, labels, 10)
        print(predictions)
elif(algorithm == 2):
    #######################Random Forest#######################
    clfRF = RandomForestClassifier()
    clfRF.fit(x_train,y_train)
    predictions = clfRF.predict(x_test)
    print("#######################Random Forest#######################")
    print_stats_metrics(y_test, predictions)
    crossValidation = input("Run 10-k Fold?\nY/N: ")
    if(crossValidation == 'Y'):
        predictions = cross_validation(clfRF, feature_std, labels, 10)
        print(predictions)
elif(algorithm == 3):
    ####################### KNN #######################
    clfKNN = KNeighborsClassifier()
    clfKNN.fit(x_train, y_train)
    predictions = clfKNN.predict(x_test)
    print("####################### KNN #######################")
    print_stats_metrics(y_test, predictions)
    crossValidation = input("Run 10-k Fold?\nY/N: ")
    if(crossValidation == 'Y'):
        predictions = cross_validation(clfKNN, feature_std, labels, 10)
        print(predictions)
elif(algorithm == 4):
    ####################### Decision Tree #######################
    clfDTC = DecisionTreeClassifier()
    clfDTC.fit(x_train, y_train)
    predictions = clfDTC.predict(x_test)
    print("####################### Decision Tree #######################")
    print_stats_metrics(y_test, predictions)
    crossValidation = input("Run 10-k Fold?\nY/N: ")
    if(crossValidation == 'Y'):
        predictions = cross_validation(clfDTC, feature_std, labels, 10)
        print(predictions)





