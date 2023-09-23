"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

# RF & DT Cross-Validation ====================== Behzad Amanpour ===================
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

model1 = RandomForestClassifier(random_state=21)
model2 = DecisionTreeClassifier(random_state=21)

scores = cross_val_score(model1, X, y, cv=5, scoring='f1')  # scoring could be 'recall', 'precision', 'f1', ...
print("model1 f1 score:", np.mean(scores))
scores = cross_val_score(model2, X, y, cv=5, scoring='f1')
print("model2 f1 score:", np.mean(scores))

# Optimization (Tuning) ========================= Behzad Amanpour ===================
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [10, 50, 100, 200]}   # the number of estimators (trees) is the most important parameter in RF
model3 = GridSearchCV(model1, param_grid, scoring='f1', cv=5)
model3.fit(X, y)
print("best params:", model3.best_params_)
