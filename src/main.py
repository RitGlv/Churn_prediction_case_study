import pandas as pd
import numpy as np
import clean_data as cln
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

reload (cln)

def prepare_data(x_split,logistic = False):
    '''
        clean the data, if logistic regression needed then constant_and_drop=True
    '''
    X = cln.clean_all(x_split,constant_and_drop=logistic)
    if logistic:
        X.pop('churn')
        return X
    y = X.pop('churn')

    return X,y

def create_estimators():
    '''
    create estimators: random forest, decision tree, adaboost, GradientBoosting, SVM, LogisticRegression with lasso regularization
    '''
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    ada = AdaBoostClassifier(DecisionTreeClassifier())
    gb = GradientBoostingClassifier()
    knn = KNeighborsClassifier()
    svc = SVC()
    lr = LogisticRegression(penalty='l1')

    return {'Random_Forest':rf,'Decision_Tree':dt,'AdaBoost':ada,'GradientBoosting':gb,'KNeighbors':knn,'SVC':svc,'Logistic_Regression':lr}

if __name__=="__main__":

    df = pd.read_csv('data/churn_train.csv')
    X_train, X_test = train_test_split(df, test_size = 0.2, random_state = 1)

    X,y = prepare_data(X_train)
    X_log = prepare_data(X_train,logistic=True)
    estimators = create_estimators()
