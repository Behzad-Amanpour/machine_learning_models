"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which is a continuous variable (like blood pressure) or discrete variable with many values (like age = 0, 1, 2, ..., 100)

for 'Regression with Statsmodels', please scroll down to the end of the page
"""

# Regression with sklearn ================= Behzad Amanpour ========================
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y) # fit_intercept=True by default
model.score( X, y)
coef = model.coef_ # you can see the coef in the variable Ecplorer of your IDE (e.g. Spyder)
print ( model.coef_)
print ( model.intercept_ )
y_pred = model.predict( X )

from sklearn.metrics import r2_score, mean_squared_error

r2_score(y, y_pred)  # equal to model.score( X, y)
mean_squared_error(y, y_pred)

# Standardization ========================= Behzad Amanpour ===================
from scipy.stats import zscore
import copy

X2 = copy.copy(X) # I use "copy" because if you change "X", "X2" will not be chnaged
X = zscore( X) # zscore(X, axis=1) calculates zscore in rows
model = LinearRegression()
model.fit(X, y) # fit_intercept=True by default
model.score( X, y)
coef2 = model.coef_
print ( model.coef_)

# Normalization =========================== Behzad Amanpour ===================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
model = LinearRegression()
model.fit(X, y) # fit_intercept=True by default
model.score( X, y)
coef3 = model.coef_
print ( model.coef_)

# Polynomial features ====================== Behzad Amanpour ===========================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

X = copy.copy(X2) # "X" has been changed, and is now restored from "X2"
degrees = [1,2,3,4,5,6,7,8,9] # range(1,10)
for i in range(len(degrees)):
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    model = LinearRegression()
    model2 = Pipeline(
        [   
            ("polynomial_features", polynomial_features),
            ("linear_regression", model),
        ])
    model2.fit(X, y)
    print('degree: ',degrees[i])
    print('numebr of features: ',model.n_features_in_)       
    print('r2_score:',model2.score( X, y))

polynomial_features = PolynomialFeatures(degree=5, include_bias=False)
model = LinearRegression()
model2 = Pipeline(
    [   
        ("polynomial_features", polynomial_features),
        ("linear_regression", model),
    ])
model2.fit(X, y)
y_pred = model2.predict( X )

# Overfitting ============================== Behzad Amanpour ==============================
from sklearn.model_selection import cross_val_score
import numpy as np

degrees = [1,2,3,4,5,6,7,8,9]  # the higher the dثgree, the higher the probability of overfitting
for i in range(len(degrees)):
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    model = LinearRegression()
    pipeline = Pipeline(
        [   
            ("polynomial_features", polynomial_features),
            ("linear_regression", model),
        ])
    print('degree: ',degrees[i])
    scores = cross_val_score( pipeline, X, y, scoring="r2", cv=3   )
    print('r2_score:',np.mean(scores))

# Statsmodels.api =========================== Behzad Amanpour ============================
!pip install statsmodels
import statsmodels.api as sm
import numpy as np

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.rsquared)
print(results.summary())
