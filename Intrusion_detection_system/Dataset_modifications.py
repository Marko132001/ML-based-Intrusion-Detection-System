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
from sklearn.model_selection import StratifiedKFold, KFold

simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=np.inf)

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

df = pd.read_csv('NF-BoT-IoT.csv', header=0, dtype=float, usecols=range(4, 12))
dt = pd.read_csv('NF-BoT-IoT.csv', header=0, dtype=int, usecols=[12])


####Scale feature matrix######
feature_std = StandardScaler().fit_transform(df.values)

labels = LabelEncoder().fit_transform(dt.values)


x_train, x_test, y_train, y_test = train_test_split(feature_std, labels, test_size=0.34, random_state=42, shuffle=True)


print("Begin:__________________________________")

def print_stats_metrics(y_test, y_pred, y_pred_prob):    
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
    print('AUC score: %.3f' % roc_auc_score(y_test, y_pred_prob))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % auc(fpr, tpr))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
      

algorithm = int(input("Choose algorithm:\n1-Naive Bayes\n2-Random Fores\n3-KNN\n4-Decision Tree\nEnter number: "))

if(algorithm == 1):
    #######################Naive Bayes#######################
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("#######################Naive Bayes#######################")
    print_stats_metrics(y_test, predictions, pred_prob)

elif(algorithm == 2):
    #######################Random Forest#######################
    clf = RandomForestClassifier()
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("#######################Random Forest#######################")
    print_stats_metrics(y_test, predictions, pred_prob)

elif(algorithm == 3):
    ####################### KNN #######################
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("####################### KNN #######################")
    print_stats_metrics(y_test, predictions, pred_prob)

elif(algorithm == 4):
    ####################### Decision Tree #######################
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("####################### Decision Tree #######################")
    print_stats_metrics(y_test, predictions, pred_prob)
    

crossValidation = input("Run 10-k Fold?\nY/N: ")

if(crossValidation == 'Y'):
    stratKFold = input("Run stratisfied KFold?\nY/N: ")
    metric = int(input("Select metric:\n1-Accuracy\n2-Precision\n3-Recall\n4-F1 score\n5-ROC-AUC score\n6-Geometric mean\nEnter number: "))

    naive_bayes = []
    random_forest = []
    knn = []
    decision_tree = []
    clfNB = GaussianNB()
    clfRF = RandomForestClassifier()
    clfKNN = KNeighborsClassifier()
    clfDT = DecisionTreeClassifier()
    
    if(stratKFold == 'N'):
        kf = KFold(n_splits=10, random_state=1, shuffle=True)
    elif(stratKFold == 'Y'):
        kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    for train_index, test_index in kf.split(df.values, dt.values):
        X_train = feature_std[train_index]
        y_train = labels[train_index]
        X_test = feature_std[test_index]
        y_test = labels[test_index]
        
        clfNB.fit(X_train,y_train)
        predictNB_prob = clfNB.predict_proba(X_test)[:, 1]
        predictNB = clfNB.predict(X_test)
        
        clfRF.fit(X_train, y_train)
        predictRF_prob = clfRF.predict_proba(X_test)[:, 1]
        predictRF = clfRF.predict(X_test)

        clfKNN.fit(X_train, y_train)
        predictKNN_prob = clfKNN.predict_proba(X_test)[:, 1]
        predictKNN = clfKNN.predict(X_test)

        clfDT.fit(X_train, y_train)
        predictDT_prob = clfDT.predict_proba(X_test)[:, 1]
        predictDT = clfDT.predict(X_test)

        match metric:
            case 1:
                naive_bayes.append(accuracy_score(y_test, predictNB))
                random_forest.append(accuracy_score(y_test, predictRF))
                knn.append(accuracy_score(y_test, predictKNN))
                decision_tree.append(accuracy_score(y_test, predictDT))
                metricStr = "Accuracy"
            case 2:
                naive_bayes.append(precision_score(y_true=y_test, y_pred=predictNB, average='binary'))
                random_forest.append(precision_score(y_true=y_test, y_pred=predictRF, average='binary'))
                knn.append(precision_score(y_true=y_test, y_pred=predictKNN, average='binary'))
                decision_tree.append(precision_score(y_true=y_test, y_pred=predictDT, average='binary'))
                metricStr = "Precision"
            case 3:
                naive_bayes.append(recall_score(y_true=y_test, y_pred=predictNB))
                random_forest.append(recall_score(y_true=y_test, y_pred=predictRF))
                knn.append(recall_score(y_true=y_test, y_pred=predictKNN))
                decision_tree.append(recall_score(y_true=y_test, y_pred=predictDT))
                metricStr = "Recall"
            case 4:
                naive_bayes.append(f1_score(y_true=y_test, y_pred=predictNB))
                random_forest.append(f1_score(y_true=y_test, y_pred=predictRF))
                knn.append(f1_score(y_true=y_test, y_pred=predictKNN))
                decision_tree.append(f1_score(y_true=y_test, y_pred=predictDT))
                metricStr = "F1 measure"
            case 5:
                naive_bayes.append(roc_auc_score(y_test, predictNB_prob))
                random_forest.append(roc_auc_score(y_test, predictRF_prob))
                knn.append(roc_auc_score(y_test, predictKNN_prob))
                decision_tree.append(roc_auc_score(y_test, predictDT_prob))
                metricStr = "ROC-AUC"
            case 6:
                tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictNB).ravel()
                tp_rate = tp/(tp+fn)
                tn_rate = tn/(tn+fp) 
                g_mean = np.sqrt(tp_rate*tn_rate)
                naive_bayes.append(g_mean)

                tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictRF).ravel()
                tp_rate = tp/(tp+fn)
                tn_rate = tn/(tn+fp) 
                g_mean = np.sqrt(tp_rate*tn_rate)
                random_forest.append(g_mean)

                tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictKNN).ravel()
                tp_rate = tp/(tp+fn)
                tn_rate = tn/(tn+fp) 
                g_mean = np.sqrt(tp_rate*tn_rate)
                knn.append(g_mean)

                tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictDT).ravel()
                tp_rate = tp/(tp+fn)
                tn_rate = tn/(tn+fp) 
                g_mean = np.sqrt(tp_rate*tn_rate)
                decision_tree.append(g_mean)
                metricStr = "Geometric mean"

        

    plt.figure()
    box_plot_data=[naive_bayes, random_forest, knn, decision_tree]
    plt.boxplot(box_plot_data,patch_artist=True,labels=['Naive Bayes', 'Random Forest', 'KNN', 'Decision Tree'])
    plt.ylim(0, 1)
    plt.ylabel(metricStr)
    plt.xlabel("Klasifikatori")
    plt.show()
        





