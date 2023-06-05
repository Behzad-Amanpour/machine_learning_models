#================================================================================================
#==================================== Classification Models =====================================
#================================================================================================

"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""


# Support Vector Machine
from sklearn.svm import SVC
model = SVC(kernel="rbf") # default is also "rbf"
model = SVC(kernel="linear")
model = SVC(kernel="poly", degree=2) # degree=3,4,5,...

# K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier as KNN
model = KNN(n_neighbors = 3)  # default is n_neighbors = 5
