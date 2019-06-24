 # -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:35:53 2018

@author: sunym
"""
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.cluster
"""
The following part is for surpervised calssification
"""
def Lasso_classifier(X_train, Y_train, X_test,**kwargs):
    clf = sklearn.linear_model.Lasso(**kwargs)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred
        
def KNN_classifier(X_train, Y_train, X_test, **kwargs):
    clf = sklearn.neighbors.KNeighborsClassifier(**kwargs)
    clf.fit(X_train, Y_train)
    # Predict
    Y_pred = clf.predict(X_test)
    return Y_pred
    
def SVM_classifier(X_train, Y_train, X_test, **kwargs):
    clf = sklearn.svm.SVC(**kwargs)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred

def SGD_classifier(X_train, Y_train, X_test, **kwargs):
    clf = sklearn.linear_model.SGDClassifier(**kwargs)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred

def DecisionTree_classifier(X_train, Y_train, X_test, **kwargs):
    clf = sklearn.tree.DecisionTreeClassifier(**kwargs)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred

def RandomForest_classifier(X_train, Y_train, X_test, **kwargs):
    clf = sklearn.ensemble.RandomForestClassifier(**kwargs)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred

def AdaBosst_classifier(X_train, Y_train, X_test, **kwargs):
    clf = sklearn.ensemble.AdaBoostClassifier(**kwargs)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred

"""
The following part is for supervised regression
"""
def lasso_regressor(X_train, Y_train, X_test, **kwargs):
    reg = sklearn.linear_model.Lasso(**kwargs)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    return Y_pred

def linear_regressor(X_train, Y_train, X_test, **kwargs):
    reg = sklearn.linear_model.LinearRegression(**kwargs)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    return Y_pred

def Ridge_regressor(X_train, Y_train, X_test, **kwargs):
    reg = sklearn.linear_model.Ridge(**kwargs)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    return Y_pred

def SVM_regressor(X_train, Y_train, X_test, **kwargs):
    reg = sklearn.svm.SVR(**kwargs)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    return Y_pred

def SGD_regressor(X_train, Y_train, X_test, **kwargs):
    reg = sklearn.linear_model.SGDRegressor(**kwargs)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    return Y_pred

def KNN_regressor(X_train, Y_train, X_test, **kwargs):
    reg = sklearn.neighbors.KNeighborsRegressor(**kwargs)
    reg.fit(X_train, Y_train)
    # Predict
    Y_pred = reg.predict(X_test)
    return Y_pred

def DecisionTree_regressor(X_train, Y_train, X_test, **kwargs):
    reg = sklearn.tree.DecisionTreeRegressor(**kwargs)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    return Y_pred

"""
The following part is for unsurpervised clustering
"""
def kmeans_clustering(X_train, X_test, **kwargs):
    kmeans = sklearn.cluster.KMeans(random_state=0,**kwargs ).fit(X_train)
    Y_pred = kmeans.predict(X_test)
    return Y_pred

def AffinityPropagation(X_train, X_test, **kwargs):
    AP = sklearn.cluster.AffinityPropagation(**kwargs).fit(X_train)
    Y_pred = AP.predict(X_test)
    return Y_pred

def AgglomerativeClustering(X_train,X_test,**kwargs):
    Agg = sklearn.cluster.AgglomerativeClustering().fit(X_train)
    Y_pred = Agg.fit_predict(X_test)
    return Y_pred

"""
The following part is for dimensionality reduction methods
"""
def PCA_DR(X_train, X_test=None, Y_train=None, **kwargs):
    clf = sklearn.decomposition.PCA(**kwargs)
    Z = clf.fit_transform(X_train)
    return Z
        


"""
heri
spectrum
"""