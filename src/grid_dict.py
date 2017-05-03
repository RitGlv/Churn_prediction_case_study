
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def grid_serach_dict():
    gd_boost = {'learning_rate':[1,0.05,0.02,0.01],
        'max_depth':[2,4,6],
        'max_features':['sqrt', 'log2'],
        'n_estimators':[50, 100, 1000]}

    ada_boost = {
        'learning_rate':[1,0.05,0.02,0.01],
        'base_estimator__max_depth':[2,4,6],
        'base_estimator__max_features':['sqrt', 'log2'],
        'n_estimators':[50, 100, 1000]}

    decision_tree = {'max_depth':[2,4,6,10],
        'min_samples_split':[5,10,20],
        'min_samples_leaf':[3,5,9,17]}

    random_forest_grid = {
        'n_estimators': [50, 100, 1000],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2, 10, 50],
        }

    knn_grid = {
        'n_neighbors': [5, 10, 15],
        'weights': ['uniform', 'distance'],
        }

    return [
        (GradientBoostingClassifier(),gd_boost),
        (AdaBoostClassifier(DecisionTreeClassifier()),ada_boost),
        (DecisionTreeClassifier(),decision_tree),
        (RandomForestClassifier(), random_forest_grid),
        (KNeighborsClassifier(), knn_grid)
    ]
