"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which is a continuous variable (like blood pressure) or discrete variable with many values (like age = 0, 1, 2, ..., 100)
"""

from scipy.stats import zscore
X = zscore( X) # zscore(X, axis=1) calculates zscore in rows
Regression_result = {}

# Bagging =================== Behzad Amanpour ===============================
from sklearn.ensemble import BaggingRegressor

model = BaggingRegressor(estimator = LinearRegression()
                          n_estimators=100,
                          # bootstrap_features = True,
                          random_state=6)
scores = cross_val_score(model, X, y, cv=4, scoring="r2")
Regression_result[ 'Bagging' ] = [scores, scores.mean(), scores.std()]

# Voting =================== Behzad Amanpour ===============================
from sklearn.ensemble import VotingRegressor

from sklearn.linear_model import LinearRegression as LR
model1 = LR()

from sklearn.neighbors import KNeighborsRegressor as knn
model2 = knn() # knn(n_neighbors = 3)

from sklearn import svm
model3 = svm.SVR(kernel="rbf")

from sklearn import svm
model4 = svm.SVR(kernel="linear")

from sklearn.ensemble import RandomForestRegressor as RF
model5 = RF(random_state=8)

estimators = [ ('LR', model1), ('KNN', model2), ('SVR_RBF', model3),
               ('SVR_Li', model4), ('RF', model5) ]

model = VotingRegressor(estimators= estimators)
                         
scores = cross_val_score(model, X, y, cv=4, scoring="r2")                       

Regression_result[ 'Voting' ] = [scores, scores.mean(), scores.std()]

# Stacking =================== Behzad Amanpour ===============================
from sklearn.ensemble import StackingRegressor

from sklearn.linear_model import LinearRegression as LR
model1 = LR()

from sklearn.neighbors import KNeighborsRegressor as knn
model2 = knn() # knn(n_neighbors = 3)

from sklearn import svm
model3 = svm.SVR(kernel="rbf")

from sklearn import svm
model4 = svm.SVR(kernel="linear")

from sklearn.ensemble import RandomForestRegressor as RF
model5 = RF(random_state=8)

estimators = [ ('LR', model1), ('KNN', model2),
               ('SVM_RBF', model3), ('SVM_Li', model4)]

model = StackingRegressor( estimators = estimators, final_estimator = model5 )
                          
scores = cross_val_score(model, X, y, cv=4, scoring='r2')                        

Regression_result[ 'Stacking' ] = [scores, scores.mean(), scores.std()]

# AdaBoost =================== Behzad Amanpour ===============================
from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor(random_state=0, n_estimators=100)

scores = cross_val_score(model, X, y, cv=4, scoring="r2")                       

Regression_result[ 'AdaBoost' ] = [scores, scores.mean(), scores.std()]

# XGBoost ======================= Behzad Amanpour =============================
import xgboost as xgb

model = xgb.XGBRegressor()
        # base_score=0.5, booster='gbtree', colsample_bylevel=1,
        #        colsample_bynode=1, colsample_bytree=1, gamma=0,
        #        importance_type='gain', learning_rate=0.1, max_delta_step=0,
        #        max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
        #        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
        #        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
        #        silent=None, subsample=1, verbosity=1

scores = cross_val_score(model, X, y, cv=4, scoring="r2")                       

Regression_result[ 'XGBoost' ] = [scores, scores.mean(), scores.std()]
