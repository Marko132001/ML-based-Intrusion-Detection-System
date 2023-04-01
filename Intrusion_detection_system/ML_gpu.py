from warnings import simplefilter

import cudf
import cupy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cuml.preprocessing import LabelEncoder, StandardScaler
from cuml.model_selection import train_test_split
from cuml.naive_bayes import GaussianNB
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.metrics import roc_auc_score, accuracy_score

simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=np.inf)



df = cudf.read_csv('NF-BoT-IoT.csv', header=0, dtype=cupy.float32, usecols=range(4, 12))
dt = cudf.read_csv('NF-BoT-IoT.csv', header=0, dtype=cupy.int32, usecols=[12])

feature_std = StandardScaler().fit_transform(df.values)
labels = LabelEncoder().fit_transform(dt.values)

x_train, x_test, y_train, y_test = train_test_split(feature_std, labels, test_size=0.34, random_state=42, shuffle=True)


algorithm = int(input("Choose algorithm:\n1-Naive Bayes\n2-Random Forest\n3-KNN\nEnter number: "))

if(algorithm == 1):
    #######################Naive Bayes#######################
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("#######################Naive Bayes#######################")

elif(algorithm == 2):
    #######################Random Forest#######################
    clf = RandomForestClassifier()
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("#######################Random Forest#######################")
    cu_score = accuracy_score( y_test, predictions )
    print( " cuml accuracy: ", cu_score )

elif(algorithm == 3):
    ####################### KNN #######################
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)[:, 1]
    print("####################### KNN #######################")
    cu_score = accuracy_score( y_test, predictions )
    print( " cuml accuracy: ", cu_score )
