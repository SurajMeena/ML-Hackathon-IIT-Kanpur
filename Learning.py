#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 08:43:21 2019

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import PyQt5


df = pd.read_csv("train_file.csv")
df_test = pd.read_csv("test_file.csv")
df.UsageClass.unique()
df_test.UsageClass.unique()
df.CheckoutType.unique()
df.CheckoutYear.unique()
df.CheckoutMonth.unique()
# Observation: Only one value(Physical, Horizon, 2005) exists in UsageClass, CheckoutType,
# CheckoutYear for both training and test set so it is redundant in training our dataset.
df_test.UsageClass.unique()
df_test.CheckoutType.unique()
df_test.CheckoutYear.unique()
df_test.CheckoutMonth.unique()

df.isnull().sum()
df_test.isnull().sum()
# Observation: Creator has 23,137 mssing values out of 31,653 values, So I will not include
# it in training, similar case exists for Publisher and Publication year...
# so here subject and title are the overall determing factor here and we need to treat
# subject like we treated it in ML course

df_modified = df.drop(["CheckoutType", "CheckoutMonth", "CheckoutYear","UsageClass", "Creator", "Publisher", "PublicationYear"], axis=1)
df_test_modified = df_test.drop(["CheckoutType", "CheckoutMonth", "CheckoutYear", "UsageClass", "Creator", "Publisher", "PublicationYear"], axis=1)
df_modified.isnull().sum()
df_test_modified.isnull().sum()

null_data = df_modified[df_modified.isnull().any(axis=1)]
null_test_data = df_test_modified[df_test_modified.isnull().any(axis=1)]

df['MaterialType'].value_counts()
null_data['MaterialType'].value_counts()   #book turns out to be a dominant class
null_test_data['MaterialType'] = 'BOOK'
null_test_data['MaterialType'].value_counts()
null_test_data = null_test_data.drop(["Checkouts", "Title", "Subjects"], axis=1)

# try model with including missing subject values by predicting their class as BOOK
df_modified.info()
df_test_modified.info()

df_modified = df_modified.dropna(axis=0, how='any')
df_test_modified = df_test_modified.dropna(axis=0, how='any')
ax = sns.boxplot(x=df_modified["Checkouts"])# use orient= 'v' for vertical boxplot
#let's normalize the attribute checkouts
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train[:,0])
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#We will now do text cleaning and other things
# Cleaning the texts
import re   # library helpful for text cleaning
import nltk
from nltk.corpus import stopwords # this imports the stopwords downloaded data
from nltk.stem.porter import PorterStemmer
df_modified = df_modified.reset_index()
''' this will reset the index, otherwise we will get errors, to
understand note the index in df_modified before running this command
'''
corpus = []
for i in range(0, 29890):
    subject = re.sub('[^a-zA-Z]', ' ', df_modified['Subjects'][i]) # returns string type as output
    subject = subject.lower() # Will lowercase all alphabets in a string
    subject = subject.split() # makes a list of words from a statement(subject)
    ps = PorterStemmer()
    subject = [ps.stem(word) for word in subject if not word in set(stopwords.words('english'))]
    '''iterations in a set are faster than in a list so for large articles above code with
    stopwords as a set will run faster. Also ps.stem() when applied on word means all those
    words obtained after implementing for loop will then be stemmed'''
    subject = ' '.join(subject) # ' ' represents that join elements of list using space as a seperator
    corpus.append(subject)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) # changing it to 1500 will not change the accuracy significantly
'''maxfeatures is used to remove irrelevant words like world steve in one of the subjects'''
X = cv.fit_transform(corpus).toarray()
'''to.array() forms a matrix of X, without it no variable named X appears on variable explorer'''
sum_col = X.sum(axis=0, dtype='int') #to check count in each column of x
X_df = pd.DataFrame(X)
X_df = X_df.reindex(X_df.sum().sort_values(ascending=False).index, axis=1)
X_df.columns = range(X_df.shape[1])
sum_col = X_df.sum()
X = X_df.values

y = df_modified.iloc[:, 5].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

#Now, we will add checkouts in X
checkouts = df_modified['Checkouts']
checkouts = checkouts.values
X = np.column_stack((X, checkouts))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#let's perform feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Now's the final step to create a model, we have to decide which classification technique to use
# Fitting SVM to the Training set
from sklearn.svm import SVC
#from sklearn.model_selection import GridSearchCV
#params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
##callbacks = [cp_callback] only if you are using colab
#
## Performing CV to tune parameters for best SVM fit
#svm_model = GridSearchCV(SVC(), params_grid, cv= 10)
#svm_model.fit(X, y)

svm_model = SVC(kernel='rbf', random_state=0, gamma='auto')
svm_model.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=svm_model, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

# Predicting on splitted test set
y_pred = svm_model.predict(X_test)

#params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3], 'C':[1, 10, 100, 1000]}]
#svm_model = GridSearchCV(SVC(), params_grid, cv = 5, n_jobs = -1, scoring = 'accuracy')
#best_accuracy = svm_model.best_score_
#best_parameters = svm_model.best_params_

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (y_pred == y_test)
svm_accuracy = accuracy.sum()/5978
from sklearn.metrics import f1_score
WeightedF1 = f1_score(y_test, y_pred, average='weighted')

import pickle
# Save the trained model as a pickle string.
#saved_model = pickle.dumps(svm_model)
## Load the pickled model
#svm_from_pickle = pickle.loads(saved_model)
## Use the loaded model to make predictions
#svm_from_pickle.predict(X)
# Use the loaded pickled model to make predictions
#svm_from_pickle.predict(X)

#import joblib
## Save the model as a pickle in a file
#joblib.dump(svm_model, 'loadModel.pkl')
#
## Load the model from the file
#svm_from_joblib = joblib.load('loadModel.pkl')
#
## Use the loaded model to make predictions
#svm_from_joblib.predict(X)

#import pickle
# save the model to disk
filename = 'FinalModel.sav'
pickle.dump(svm_model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)

#going for predictions on test set
df_test_modified = df_test_modified.reset_index()
''' this will reset the index, otherwise we will get errors, to
understand note the index in df_modified before running this command
'''
corpus1 = []
for i in range(0, 19889):
    subject = re.sub('[^a-zA-Z]', ' ', df_test_modified['Subjects'][i]) # returns string type as output
    subject = subject.lower() # Will lowercase all alphabets in a string
    subject = subject.split() # makes a list of words from a statement(subject)
    ps = PorterStemmer()
    subject = [ps.stem(word) for word in subject if not word in set(stopwords.words('english'))]
    '''iterations in a set are faster than in a list so for large articles above code with
    stopwords as a set will run faster. Also ps.stem() when applied on word means all those
    words obtained after implementing for loop will then be stemmed'''
    subject = ' '.join(subject) # ' ' represents that join elements of list using space as a seperator
    corpus1.append(subject)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1000)
'''maxfeatures is used to remove irrelevant words like world steve in one of the subjects'''
test_set = cv.fit_transform(corpus1).toarray()
'''to.array() forms a matrix of X, without it no variable named X appears on variable explorer'''

#Now, we will add checkouts in X
checkouts = df_test_modified['Checkouts']
checkouts = checkouts.values
test_set = np.column_stack((test_set, checkouts))

#let's perform feature scaling
test_set = scaler.transform(test_set)

# Finally let's make predictions on our actual test set
y_test_preds = loaded_model.predict(test_set)
y_test_preds = labelencoder.inverse_transform(y_test_preds)
#Preparing to write a CSV file
submission_format = df_test_modified.drop(["index", "Subjects", "Checkouts","Title"], axis=1)
submission_format['MaterialType'] = y_test_preds
FinalSubmission = pd.concat([submission_format, null_test_data])
FinalSubmission = FinalSubmission.sort_values(by='ID')
FinalSubmission.to_csv("/home/suraj/Desktop/ML Problems/ML hackathon Problem/FinalSubmission.csv", header=True, index=None)
