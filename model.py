import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('dataset.csv')

x = df.iloc[:, :-1]
y = df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=100)

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
],
remainder='passthrough')

ra_pipe = Pipeline([
    ('step1',trf),
    ('step2',RandomForestClassifier())
])

ra_pipe.fit(x_train,y_train)

ra_y_pred = ra_pipe.predict(x_test)

pickle.dump(ra_pipe,open('ra_pipe.pkl','wb'))

print(accuracy_score(y_test,ra_y_pred))