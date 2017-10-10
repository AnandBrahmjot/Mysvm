import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt

data = pd.read_csv('/home/mandeep/bRAHM/SVMwork/regressData.csv', header=0,index_col='Date',
                       names=['Date', 'Open', 'High','Low', 'Volume', 'Close','Close1'])

# print(data.head())
data=data.values
X = data[:-10, 1:-2]
y= data[:-10, -2]


clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X, y)

print('Predicted',clf.predict(data[-10,1:-2]))
print('Actual',data[-10,-1])