from sklearn import svm

import pandas as pd


data=pd.read_csv('/home/mandeep/bRAHM/SVMwork/pima.csv')

data=data.values

X=data[ 0:-10,:-1]
y=data[ 0:-10,-1]

clf = svm.SVC()

clf = svm.SVC(gamma=0.001, C=100)

clf.fit(X,y)

print('Predicted',clf.predict(data[-10,:-1]))

print("Actual Prediction [",data[-10,-1],']')

