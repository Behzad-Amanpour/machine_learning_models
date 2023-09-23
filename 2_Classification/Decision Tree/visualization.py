# DT Visualization ------------ Behzad Amanpour -------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt

model = DecisionTreeClassifier(random_state=21)
model.fit(X, y)

fig = plt.figure(figsize=(18,18))
tree.plot_tree(model, 
               feature_names=['Age', 'Height', 'Weight', 'Systolic',
                              'Diastolic', 'Gender_Male','Status'],
               class_names=np.array(['non-smoker', 'smoker']),
               filled=True)
