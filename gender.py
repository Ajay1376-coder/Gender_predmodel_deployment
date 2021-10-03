import pandas as pd
#import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/AJAY SINGH/Downloads/StudentsPerformance.csv")

print(data.isnull().sum())
print(data)
print(data.duplicated().sum())
a={'male':1,'female':0}
data['gender']=data['gender'].map(a)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['race/ethnicity']=le.fit_transform(data['race/ethnicity'])
print(data)
data['math_score']=data['math score']
data['reading_score']=data['reading score']
data['writing_score']=data['writing score']

data.drop(['math score','reading score','writing score'],axis=1,inplace=True)

data['parental level of education'].value_counts()
data['parental level of education']=le.fit_transform(data['parental level of education'])
data['lunch']=le.fit_transform(data['lunch'])
data['test preparation course']=le.fit_transform(data['test preparation course'])

sns.boxplot(data['gender'])
sns.boxplot(data['race/ethnicity'])
sns.boxplot(data['math_score'])
sns.boxplot(data['reading_score'])
sns.boxplot(data['writing_score'])

#data['Total_score']=data['math score']+data['reading score']+data['writing score']

from sklearn.preprocessing import  MinMaxScaler

col=['math_score','reading_score','writing_score']
features=data[col]
features

mi=MinMaxScaler()
data[col]=mi.fit_transform(features)


sns.countplot(data['race/ethnicity'],hue=data['gender'])

#sns.distplot(data['Total_score'])

#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


plt.figure(figsize=(8,8))
sns.heatmap(data.corr(),annot=True)

X=data[['math_score','reading_score','writing_score']]
y=data['gender']

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr=RandomForestClassifier(criterion='entropy')
lr.fit(X_train,y_train)
#y_pred1=model1.predict(X_test)

import pickle

pickle.dump(lr,open('gender_pred.pkl','wb'))

gender_pred=pickle.load(open('gender_pred.pkl','rb'))























