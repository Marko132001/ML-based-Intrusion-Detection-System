from warnings import simplefilter

import pickle
import time
import shap
import cudf
import cuml
import cupy
from cuml.explainer import KernelExplainer, TreeExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, roc_curve, auc, precision_score, 
                             recall_score, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve,
                             matthews_corrcoef, cohen_kappa_score)
from sklearn.model_selection import StratifiedKFold, KFold
from cuml.preprocessing import LabelEncoder, StandardScaler
from cuml.model_selection import train_test_split
from cuml.naive_bayes import GaussianNB
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
import bctools as bc

simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=np.inf)

cuml.set_global_output_type('cupy')

dataset_input = int(input("Choose dataset:\n1-NF-BoT-IoT\n2-NF-ToN-IoT\n3-NF-UNSW-NB15\n4-NF-ToN-IoT-v2\n5-NF-UNSW-NB15-v2\n"))

knn_optimal_value = 5
dataset_name = ""

match dataset_input:
    case 1:
        dataset_name = "NF-BoT-IoT"
        knn_optimal_value = 5
    case 2:
        dataset_name = "NF-ToN-IoT"
        knn_optimal_value = 17
    case 3:
        dataset_name = "NF-UNSW-NB15"
        knn_optimal_value = 9
    case 4:
        dataset_name = "NF-ToN-IoT-v2"
    case 5:
        dataset_name = "NF-UNSW-NB15-v2"
        knn_optimal_value = 3

if(dataset_input <= 3):
    df = cudf.read_csv(f"./datasets/{dataset_name}.csv", header=0, dtype=cupy.float32, usecols=range(4, 12))
    dt = cudf.read_csv(f"./datasets/{dataset_name}.csv", header=0, dtype=cupy.int32, usecols=[12])
elif(dataset_input == 6):
    df = cudf.read_csv(f"./datasets/{dataset_name}.csv", header=0, dtype=cupy.float32, usecols=range(0, 83))
    dt = cudf.read_csv(f"./datasets/{dataset_name}.csv", header=0, dtype=cupy.int32, usecols=[83])
else:
    features = []
    nums = [16,17,41]
    for i in range(4, 43):
        if(i not in nums):
            features.append(i)
    df = cudf.read_csv(f"./datasets/{dataset_name}.csv", header=0, dtype=cupy.float32, usecols=features)
    dt = cudf.read_csv(f"./datasets/{dataset_name}.csv", header=0, dtype=cupy.int32, usecols=[43])


feature_std = StandardScaler().fit_transform(df)
labels = LabelEncoder().fit_transform(dt.values)

x_train, x_test, y_train, y_test = train_test_split(feature_std, labels, test_size=0.34, random_state=42, shuffle=True)


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
    print('Cohens kappa: %.3f' % cohen_kappa_score(y_test, y_pred))
    print('Matthews Corr. Coef.: %.3f' % matthews_corrcoef(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % auc(fpr, tpr))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    gmeans_improved = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmeans_improved)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[index], gmeans_improved[index]))

    precision, recall, thresholds_f1 = precision_recall_curve(y_test, y_pred_prob)

    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds_f1[ix], fscore[ix]))

    return thresholds[index]




def KFoldCompareMetrics(algName, customThresh=0.5):

    accuracy = []
    precision = []
    recall = []
    f1_measure = []
    geometric_mean = []

    stratKFold = input("Run stratisfied KFold?\nY/N: ")

    if(stratKFold == 'N'):
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
    elif(stratKFold == 'Y'):
        kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)


    for train_index, test_index in kf.split(df.values.get(), dt.values.get()):
        X_train = feature_std[train_index]
        y_train = labels[train_index]
        y_train = y_train.values
        X_test = feature_std[test_index]
        y_test = labels[test_index]
        y_test = y_test.values.get()

        if(algName == "Naive Bayes"):
            clf = GaussianNB()
        elif(algName == "KNN"):
            clf = KNeighborsClassifier(n_neighbors=knn_optimal_value)
        elif(algName == "Random Forest"):
            clf = RandomForestClassifier()
        elif(algName == "Decision Tree"):
            clf = RandomForestClassifier(n_estimators=1, bootstrap=False)

        clf.fit(X_train,y_train)

        if(customThresh == 0.5):
            #predict_prob = clf.predict_proba(X_test)[:, 1].get()
            predict = clf.predict(X_test).get()
        else:
            pred_prob = clf.predict_proba(X_test)[:, 1].get()
            predict = []
            for val in pred_prob:
                if(val < new_threshold):
                    predict.append(0)
                else:
                    predict.append(1)

        accuracy.append(accuracy_score(y_test, predict))
        precision.append(precision_score(y_true=y_test, y_pred=predict, average='binary'))
        recall.append(recall_score(y_true=y_test, y_pred=predict))
        f1_measure.append(f1_score(y_true=y_test, y_pred=predict))
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predict).ravel()
        tp_rate = tp/(tp+fn)
        tn_rate = tn/(tn+fp) 
        g_mean = np.sqrt(tp_rate*tn_rate)
        geometric_mean.append(g_mean)

    plt.figure()
    box_plot_data=[accuracy, precision, recall, f1_measure]
    plt.boxplot(box_plot_data,patch_artist=True,labels=['Accuracy', 'Precision', 'Recall', 'F1 Measure'])
    plt.ylim(0, 1)
    plt.ylabel(algName)
    plt.xlabel("Klasifikacijske metrike")
    plt.show()


def KFoldCompareAlgorithms():

    stratKFold = input("Run stratisfied KFold?\nY/N: ")
    metric = int(input("Select metric:\n1-Accuracy\n2-Precision\n3-Recall\n4-F1 score\n5-ROC-AUC score\n6-Geometric mean\n7-MCC\nEnter number: "))

    naive_bayes = []
    random_forest = []
    knn = []
    decision_tree = []
    
    if(stratKFold == 'N'):
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
    elif(stratKFold == 'Y'):
        kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(df.values.get(), dt.values.get()):
        X_train = feature_std[train_index]
        y_train = labels[train_index]
        y_train = y_train.values
        X_test = feature_std[test_index]
        y_test = labels[test_index]
        y_test = y_test.values.get()

        clfNB = GaussianNB()
        clfRF = RandomForestClassifier()
        clfKNN = KNeighborsClassifier(n_neighbors=knn_optimal_value)
        clfDT = RandomForestClassifier(n_estimators=1, bootstrap=False)
        
        clfNB.fit(X_train,y_train)
        predictNB_prob = clfNB.predict_proba(X_test)[:, 1].get()
        predictNB = clfNB.predict(X_test).get()
        
        clfRF.fit(X_train, y_train)
        predictRF_prob = clfRF.predict_proba(X_test)[:, 1].get()
        predictRF = clfRF.predict(X_test).get()

        clfKNN.fit(X_train, y_train)
        predictKNN_prob = clfKNN.predict_proba(X_test)[:, 1].get()
        predictKNN = clfKNN.predict(X_test).get()

        clfDT.fit(X_train, y_train)
        predictDT_prob = clfDT.predict_proba(X_test)[:, 1].get()
        predictDT = clfDT.predict(X_test).get()

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
            case 7:
                naive_bayes.append(matthews_corrcoef(y_true=y_test, y_pred=predictNB))
                random_forest.append(matthews_corrcoef(y_true=y_test, y_pred=predictRF))
                knn.append(matthews_corrcoef(y_true=y_test, y_pred=predictKNN))
                decision_tree.append(matthews_corrcoef(y_true=y_test, y_pred=predictDT))
                metricStr = "MCC"

    if(metric == 7):
        y_low = -1
        y_high = 1
    else:
        y_low = 0
        y_high = 1

    plt.figure()
    box_plot_data=[naive_bayes, random_forest, knn, decision_tree]
    plt.boxplot(box_plot_data,patch_artist=True,labels=['Naive Bayes', 'Random Forest', 'KNN', 'Decision Tree'])
    plt.ylim(y_low, y_high)
    plt.ylabel(metricStr)
    plt.xlabel("Klasifikatori")
    plt.show()


def plot_optimal_k_value():
    error_rate = []
    for i in range(1, 31, 2):
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        error_rate.append(np.mean(predictions.get() != y_test.get()))
    plt.figure(figsize=(10,6))
    plt.plot(range(1,31, 2),error_rate,color='blue', linestyle='dashed', marker='o',
    markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()



algorithm = int(input("Choose algorithm:\n1-Naive Bayes\n2-Random Forest\n3-KNN\n4-Decision Tree\nEnter number: "))



if(algorithm == 1):
    #######################Naive Bayes#######################
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    start = time.time()
    predictions = clf.predict(x_test)
    end = time.time()
    print(end - start)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("#######################Naive Bayes#######################")
    new_threshold = print_stats_metrics(y_test.get(), predictions.get(), pred_prob.get())
    kfold = input("Run KFold?\nY/N: ")
    if(kfold == 'Y'):
        optimalThresh = input("Run on optimal threshold?\nY/N: ")
        if(optimalThresh == 'N'):
            KFoldCompareMetrics("Naive Bayes")
        else:
            KFoldCompareMetrics("Naive Bayes", new_threshold)
elif(algorithm == 2):
    #######################Random Forest#######################
    clf = RandomForestClassifier()
    clf.fit(x_train,y_train)
    start = time.time()
    predictions = clf.predict(x_test)
    end = time.time()
    print(end - start)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("#######################Random Forest#######################")
    new_threshold = print_stats_metrics(y_test.get(), predictions.get(), pred_prob.get())
    kfold = input("Run KFold?\nY/N: ")
    if(kfold == 'Y'):
        optimalThresh = input("Run on optimal threshold?\nY/N: ")
        if(optimalThresh == 'N'):
            KFoldCompareMetrics("Random Forest")
        else:
            KFoldCompareMetrics("Random Forest", new_threshold)
elif(algorithm == 3):
    ####################### KNN #######################
    #optimal_k = input("Plot optimal K value?\nY/N: ")
    #if(optimal_k == 'Y'):
    #   plot_optimal_k_value()
    clf = KNeighborsClassifier(n_neighbors=knn_optimal_value)
    clf.fit(x_train, y_train)
    start = time.time()
    predictions = clf.predict(x_test)
    end = time.time()
    print(end - start)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("####################### KNN #######################")
    new_threshold = print_stats_metrics(y_test.get(), predictions.get(), pred_prob.get())
    kfold = input("Run KFold?\nY/N: ")
    if(kfold == 'Y'):
        optimalThresh = input("Run on optimal threshold?\nY/N: ")
        if(optimalThresh == 'N'):
            KFoldCompareMetrics("KNN")
        else:
            KFoldCompareMetrics("KNN", new_threshold)
elif(algorithm == 4):
    #######################Decision Tree#######################
    clf = RandomForestClassifier(n_estimators=1, bootstrap=False)
    clf.fit(x_train,y_train)
    saveModel = input("Save model?\nY/N: ")
    if(saveModel == 'Y'):
        pickle.dump(clf, open('decTreeModel.pkl', 'wb'))
    start = time.time()
    predictions = clf.predict(x_test)
    end = time.time()
    print(end - start)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("#######################Decision Tree#######################")
    new_threshold = print_stats_metrics(y_test.get(), predictions.get(), pred_prob.get())
    kfold = input("Run KFold?\nY/N: ")
    if(kfold == 'Y'):
        optimalThresh = input("Run on optimal threshold?\nY/N: ")
        if(optimalThresh == 'N'):
            KFoldCompareMetrics("Decision Tree")
        else:
            KFoldCompareMetrics("Decision Tree", new_threshold)
    
kfold_alg = input("Run KFold on all algorithms?\nY/N: ")

if(kfold_alg == 'Y'):
    KFoldCompareAlgorithms()

testModel = input("Test saved model?\nY/N: ")

if(testModel == 'Y'):
    savedClf = pickle.load(open('decTreeModel.pkl', 'rb'))
    predictions = savedClf.predict(x_test)
    pred_prob = savedClf.predict_proba(x_test)[:, 1]
    print_stats_metrics(y_test.get(), predictions.get(), pred_prob.get())

