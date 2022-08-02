import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.express as px

sns.set_theme(color_codes=True)
sns.set_style('whitegrid')
init_notebook_mode(connected=True)
cf.go_offline()

df = pd.read_csv('Churn_Modelling.csv')

df.info
df.select_dtypes(include='object')

dummies = pd.get_dummies(df[['Geography','Gender']], drop_first=True)
df.drop(['RowNumber','CustomerId','Surname'], axis = 1, inplace=True)
df.drop(['Geography','Gender'], axis = 1, inplace=True)

df = pd.concat([df,dummies], axis=1)

X = df.drop('Exited', axis = 1)
y = df['Exited']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
X.shape
classifier = Sequential()

classifier.add(Dense(11, activation=('relu')))

classifier.add(Dense(6, activation=('relu')))


classifier.add(Dense(1, activation=('sigmoid')))

classifier.compile(optimizer='adam', loss=('binary_crossentropy'), metrics=(['accuracy']))

classifier.fit(X_train, y_train, epochs=400, batch_size=32, validation_data=(X_test,y_test))

losses = pd.DataFrame(classifier.history.history)
losses.plot()

pred = classifier.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, pred))
print(confusion_matrix(y_test,pred))

new_prediction = classifier.predict_classes(scaler.transform([[600,40,3,60000,2,1,1,50000,0,0,1]]))
new_prediction = (new_prediction > 0.5)"""


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_function():
    classifier = Sequential()

    classifier.add(Dense(11, activation=('relu')))
    classifier.add(Dense(6, activation=('relu')))

    classifier.add(Dense(1, activation=('sigmoid')))

    classifier.compile(optimizer='adam', loss=('binary_crossentropy'), metrics=(['accuracy']))
    return classifier

classifier = KerasClassifier(build_fn=build_function,epochs=400, batch_size=32)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10 )   

mean = accuracies.mean() 
print(mean*100)
print(accuracies.std())


#TUNING THE ANN
"""from sklearn.model_selection import GridSearchCV

def build_function(optimizer):
    classifier = Sequential()

    classifier.add(Dense(11, activation=('relu')))
    classifier.add(Dense(6, activation=('relu')))

    classifier.add(Dense(1, activation=('sigmoid')))

    classifier.compile(optimizer = optimizer, loss=('binary_crossentropy'), metrics=(['accuracy']))
    return classifier

classifier = KerasClassifier(build_fn=build_function)
parameters = {'epochs' : [100,200,300,400],
              'batch_size' : [25,32,64],
              'optimizer' : ['adam','rmsprop']}

grid_classifier = GridSearchCV(estimator=classifier, param_grid=parameters, cv=10, scoring=('accuracy'))
grid_classifier.fit(X_train,y_train)

best_parameter = grid_classifier.best_params_
best_score = grid_classifier.best_score_""" 

