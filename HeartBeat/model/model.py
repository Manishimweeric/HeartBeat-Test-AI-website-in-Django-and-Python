import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('heart_disease_data.csv')
#print(df.describe())
#data cleaning
# df.duplicated()
# df.drop_duplicates()
#df['target']=df['target'].fillna(df['target'].median(),inplace=True)
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)

import joblib
#joblib.dump(model,'train_heart_disease_model.joblib')
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
score= accuracy_score(y_test,y_pred)
print(score)





