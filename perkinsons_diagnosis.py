import numpy as np
import pandas as pd
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from time import time
from sklearn.cross_validation import train_test_split


import matplotlib.pyplot as plt
from sklearn import datasets

parkinson_data = pd.read_csv("parkinsons.csv")
print "student data read successfully"

clf  = GaussianNB()
clf2 = svm.SVC()
clf3 = SGDClassifier(loss = "hinge")
clf4 = GradientBoostingClassifier(n_estimators=100, learning_rate = 1.0, max_depth=1, random_state=0 )


from pandas.tools.plotting import scatter_matrix
import pylab

##plot and save the scatter plot of features
pd.scatter_matrix(parkinson_data, alpha = 0.3, figsize = (30,30), diagonal = 'kde')
pylab.savefig("scatter" + ".png")


n_patients = parkinson_data.shape[0]   ##number of patients
n_features = parkinson_data.shape[1]-1 ##number of features
n_parkinsons = parkinson_data[parkinson_data['status'] == 1].shape[0] ##persons with parkinson 
n_healthy = parkinson_data[parkinson_data['status'] == 0].shape[0] ##healthy persons
print "total number of patients:", n_patients
print "number of features:", n_features
print "number of patients with parkinson:", n_parkinsons
print "number of patients without parkinsons:", n_healthy

def train_classifier(clf, x_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    start = time()
    clf.fit(x_train, y_train)
    end = time()
    
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label=1)


def train_predict(clf, x_train, y_train, x_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(x_train))
    
    train_classifier(clf, x_train, y_train)
    
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, x_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, x_test, y_test))


# Tuning / Optimization Functions

def performance_metric(y_true, y_predict):
    error = f1_score(y_true, y_predict, pos_label=1)
    return error

def fit_model(x, y):
  
    classifier = svm.SVC()

    parameters = {'kernel':['poly', 'rbf', 'sigmoid'], 'degree':[1, 2, 3], 'C':[0.1, 1, 10]}


    f1_scorer = make_scorer(performance_metric,
                                   greater_is_better=True)

    clf = GridSearchCV(classifier,
                       param_grid=parameters,
                       scoring=f1_scorer)

    clf.fit(x, y)

    return clf

feature_cols = list(parkinson_data.columns[1:16]) + list(parkinson_data.columns[18:])
target_col  = parkinson_data.columns[17]

print "feature columns:", feature_cols
print "target columns:", target_col

x_all = parkinson_data[feature_cols]
y_all = parkinson_data[target_col]

print "feature values:", x_all.head()

num_all  = parkinson_data.shape[0]
num_train = 150
num_test  = num_all - num_train

##shuffling of data into test and training set complete!"
x_train,x_test,y_train,y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=5)
print "training set:", x_train.shape[0]
print "test set:", x_test.shape[0]

x_train_50 = x_train[:50]
y_train_50 = y_train[:50]

x_train_100 = x_train[:100]
y_train_100 = y_train[:100]

x_train_150 = x_train[:150]
y_train_150 = y_train[:150]

print "naive bayes:"
train_predict(clf,x_train_50,y_train_50,x_test,y_test)

print "Support Vector Machines:"
train_predict(clf2,x_train_50,y_train_50,x_test,y_test)

print "Stochastic Gradient Descent:"
train_predict(clf3,x_train_50,y_train_50,x_test,y_test)

print "Gradient Tree Boosting:"
train_predict(clf4,x_train_50,y_train_50,x_test,y_test)

#100 set

print "Naive Bayes:"
train_predict(clf,x_train_100,y_train_100,x_test,y_test)

print "Support Vector Machines:"
train_predict(clf2,x_train_100,y_train_100,x_test,y_test)

print "Stochastic Gradient Descent:"
train_predict(clf3,x_train_100,y_train_100,x_test,y_test)

print "Gradient Tree Boosting:"
train_predict(clf4,x_train_100,y_train_100,x_test,y_test)

#150 set

print "Naive Bayes:"
train_predict(clf,x_train_150,y_train_150,x_test,y_test)

print "Support Vector Machines:"
train_predict(clf2,x_train_150,y_train_150,x_test,y_test)

print "Stochastic Gradient Descent:"
train_predict(clf3,x_train_150,y_train_150,x_test,y_test)

print "Gradient Tree Boosting:"
train_predict(clf4,x_train_150,y_train_150,x_test,y_test)



clf2 = fit_model(x_train, y_train)
print "Successfully fit a model!"

print "The best parameters were: " 

print clf2.best_params_

start = time()
    
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf2, x_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf2, x_test, y_test))

end = time()
    
print "Tuned model in {:.4f} seconds.".format(end - start)

