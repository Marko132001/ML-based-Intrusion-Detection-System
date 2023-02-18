from warnings import simplefilter

import numpy as np
import pandas as pd
import sklearn
from numpy import genfromtxt
from numpy.lib.recfunctions import append_fields
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=np.inf)

feature = genfromtxt('NF-BoT-IoT.csv', delimiter=',', usecols=(i for i in range(0, 12)), dtype=str, skip_header=1, encoding="utf-8")

#####Append certain column (feature) to feature matrix######
#feature = np.hstack((feature, np.atleast_2d(genfromtxt('NF-BoT-IoT.csv', delimiter=',', usecols=(-1), dtype=str, skip_header=1, encoding="utf-8")).T))

#####Binary classification - usecols=(-2), for Multiclass classification - usecols=(-1) !!!change dtype#######
target = genfromtxt('NF-BoT-IoT.csv', delimiter=',', usecols=(-2), dtype=int, skip_header=1)


for i in range(len(feature[:,0])):
    feature[i,0] = feature[i,0].split('.')[-1]
    feature[i,2] = feature[i,2].split('.')[-1]

"""
#####Print feature matrix to txt file#######
write_feature = np.array2string(feature)
with open('features.txt', 'w') as f:
    f.write(write_feature)
"""

"""
#####Print label matrix to txt file#######
write_target = np.array2string(target)
with open('label.txt', 'w') as f:
    f.write(write_target)
"""


####Fit transform IP addresses###### 
#feature[:,0]=LabelEncoder().fit_transform(feature[:,0])
#feature[:,2]=LabelEncoder().fit_transform(feature[:,2])

####Fit transform Attack types (Multiclass)######
#feature[:,-1]=LabelEncoder().fit_transform(feature[:,-1])

"""
#####Convert all feature data to float######
feature = feature.astype(np.float64)

write_feature = np.array2string(feature)
with open('features.txt', 'w') as f:
    f.write(write_feature)
"""


####Scale feature matrix######
feature_std = StandardScaler().fit_transform(feature)

labels = LabelEncoder().fit_transform(target)


x_train, x_test, y_train, y_test = train_test_split(feature_std, labels, test_size=0.34, random_state=42)


print("Begin:__________________________________")
## print stats

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
    ####In the event of multiclass classification --> replace average='binary' with some other option (i.e. micro)######
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='binary'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
    print('Geometric mean: %.3f' % g_mean)
    ##Dodati geometric_mean = square_root(true_positive_rate*true_negative_rate)
    ##Mjera za osjetljivije datasetove  --> za inbalanced datasets
#######################Naive Bayes#######################
print("#######################Naive Bayes#######################")
clfNB = GaussianNB()
clfNB.fit(x_train,y_train)
predictions = clfNB.predict(x_test)
print_stats_metrics(y_test, predictions)

"""
#######################Random Forest#######################
#print("#######################Random Forest#######################")
clfRF = RandomForestClassifier()
clfRF.fit(x_train,y_train)
predictions = clfRF.predict(x_test)
print("#######################Random Forest#######################")
print_stats_metrics(y_test, predictions)


"""
