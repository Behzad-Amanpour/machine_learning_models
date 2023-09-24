"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

from sklearn.model_selection import cross_val_score
from scipy.stats import zscore
X = zscore( X) # zscore(X, axis=1) calculates zscore in rows

Classification_result = {}

# Bagging =================== Behzad Amanpour ===============================
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(estimator = SVC(kernel='linear') # or any other classification model
                          n_estimators=100,
                          # bootstrap_features = True,
                          random_state=8)
                          
scores = cross_val_score(model, X, y, cv=4, scoring='balanced_accuracy') # 'recall = sensitivity' 'precision' 'f1'                        

Classification_result[ 'Bagging' ] = [scores, scores.mean(), scores.std()]

# Voting =================== Behzad Amanpour ===============================
# from warnings import simplefilter
# simplefilter(action='ignore', category=FutureWarning)
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression as LR
model1 = LR()
from sklearn.neighbors import KNeighborsClassifier as knn
model2 = knn() # knn(n_neighbors = 3)
from sklearn import svm
model3 = svm.SVC(kernel="rbf")
from sklearn import svm
model4 = svm.SVC(kernel="linear")
from sklearn.ensemble import RandomForestClassifier as RF
model5 = RF(random_state=8)

estimators = [ ('LR', model1), ('KNN', model2), ('SVM_RBF', model3),
               ('SVM_Li', model4), ('RF', model5) ]

model = VotingClassifier(estimators= estimators, voting='hard')
scores = cross_val_score(model, X, y, cv=4, scoring='balanced_accuracy') # 'recall = sensitivity' 'precision' 'f1'                        
Classification_result[ 'Voting' ] = [scores, scores.mean(), scores.std()]

# Stacking =================== Behzad Amanpour ===============================
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression as LR
model1 = LR()
from sklearn.neighbors import KNeighborsClassifier as knn
model2 = knn() # knn(n_neighbors = 3)
from sklearn import svm
model3 = svm.SVC(kernel="rbf")
from sklearn import svm
model4 = svm.SVC(kernel="linear")
from sklearn.ensemble import RandomForestClassifier as RF
model5 = RF(random_state=10)

estimators = [ ('LR', model1), ('KNN', model2),
               ('SVM_RBF', model3), ('SVM_Li', model4)]

model = StackingClassifier( estimators = estimators, final_estimator = model5 )
scores = cross_val_score(model, X, y, cv=4, scoring='balanced_accuracy') # 'recall = sensitivity' 'precision' 'f1'                        
Classification_result[ 'Stacking' ] = [scores, scores.mean(), scores.std()]

# Adaboost =================== Behzad Amanpour ===============================
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier( n_estimators=100 )
                        # learning_rate
scores = cross_val_score(model, X, y, cv=4, scoring='balanced_accuracy') # 'recall = sensitivity' 'precision' 'f1'                        
Classification_result[ 'AdaBoost' ] = [scores, scores.mean(), scores.std()]

# XGBoost ======================= Behzad Amanpour =============================
# conda install -c conda-forge py-xgboost
# !pip install xgboost
import xgboost as xgb

model = xgb.XGBClassifier( random_state=10 )
# base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
#        max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
#        n_estimators=100, n_jobs=1, nthread=None,
#        objective='multi:softprob', random_state=0, reg_alpha=0,
#        reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
#        subsample=1, verbosity=1

scores = cross_val_score(model, X, y, cv=4, scoring='balanced_accuracy')
Classification_result[ 'XGBoost' ] = [scores, scores.mean(), scores.std()]
