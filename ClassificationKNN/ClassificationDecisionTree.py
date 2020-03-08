import inline as inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import style
style.use('seaborn')

data=pd.read_csv(r'C:\Users\Sargis\AppData\Local\Programs\Python\Python38-32\kc_house_data.csv',encoding='latin')
print(data.T)
print(data['price'].describe())
res=len(data[data['price']>7e5])/len(data)
print(res)
print(data.shape)


# data['price'].plot(kind='hist',bins=60)
# plt.show()
print(data.info())
pd.set_option('display.max_colwidth', -1)
pd.read_csv(r'C:\Users\Sargis\AppData\Local\Programs\Python\Python38-32\kc_house_data.csv',
            delimiter=';',
            encoding='latin')
print(data.T)
# data.zipcode.value_counts().plot(kind='bar')
# plt.show()

uniq=data[['bedrooms',
            'bathrooms',
            'floors',
            'view',
            'condition',
            'grade',
            'yr_built',
            'yr_renovated',
          ]].apply(lambda x: x.nunique(), axis=0)
# print(uniq)

data = data.drop(['id', 'zipcode', 'date', 'sqft_basement'], axis=1, inplace=False)
print(data.shape)


print(sum(data.price>=7e5))
print(sum(data.price>=7e5)/data.shape[0])
data['high_price']=(data.price>7e5).astype('int64')
data.drop(['price'], axis=1, inplace=True)
print(len(data))
print(data.high_price.value_counts())


from sklearn import preprocessing
print(data.mean(axis=0).round(3))
print(data.std(axis=0).round(3))
data.iloc[:,:-1]= preprocessing.scale(data.iloc[:,:-1])
print(data.mean(axis=0).round(3))
print(data.std(axis=0).round(3))
print(data.T)

X,y = data.loc[:,data.columns != 'high_price'], data.loc[:,'high_price']

from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 42)


from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
DecisionTree_score=dt.score(x_test,y_test)
print("Accuracy of decision tree:",dt.score(x_test,y_test))
print(precision_score(y_pred,y_test))
print(recall_score(y_pred,y_test))


from sklearn.model_selection import train_test_split
train_data,test_data,train_y,test_y=train_test_split(data.drop('high_price', axis=1),
                            data['high_price'],
                            test_size=0.2,
                            random_state=15)


print(train_y.value_counts(normalize=True))
print(test_y.value_counts(normalize=True))


import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix,roc_auc_score


dTree = DecisionTreeClassifier()
print(dTree.fit(train_data,train_y))
pred=dTree.predict(test_data)
print(pred)

print(metrics.classification_report(test_y,pred))

from sklearn.metrics import roc_curve, auc,f1_score
fpr, tpr, _ = roc_curve(test_y, pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',)
plt.plot(np.linspace(0,1,20),np.linspace(0,1,20), '-.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
print('ROC curve (area = %0.2f)' % roc_auc)
plt.show()


