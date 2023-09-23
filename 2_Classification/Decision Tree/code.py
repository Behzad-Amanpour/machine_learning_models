"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

# DT Cross-validation =========================== Behzad Amanpour ==========================
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')   # scoring could be 'recall', 'precision', 'f1', ... 
print("cross-val accuracy:", np.mean(scores))

# Regularization (Pruning) ====================== Behzad Amanpour ==========================
model = DecisionTreeClassifier()
model.fit(X, y)
print( model.get_depth() )  
print( model.get_n_leaves() )
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)  
                                  # "max_depth" is based on "model.get_depth"
                                  # "max_leaf_nodes" is based on "model.get_n_leaves()"
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("cross-val accuracy:", np.mean(scores))

# Optimization (Grid Search) ==================== Behzad Amanpour ========================== 
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [2, 3, 5],
    'min_samples_leaf': [1, 2, 5],
    'max_leaf_nodes': [5, 10, 20]}
model = DecisionTreeClassifier()
gs = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
gs.fit(X, y)
print("best params:", gs.best_params_)
print("best score:", gs.best_score_)
