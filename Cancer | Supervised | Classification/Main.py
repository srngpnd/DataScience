from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
print('Cancer dataset keys: {}'.format(cancer.keys()))
print('Shape of the dataset: {}'.format(cancer.data.shape))
print('Count per target: {}'.format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state= 42)

training_accuracy = []
testing_accuracy = []

nearest_neighbors = range(1,11)

# KNN Classifier
for k in nearest_neighbors:
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train, y_train)

    training_accuracy.append(knn.score(X_train, y_train))
    testing_accuracy.append(knn.score(X_test, y_test))

plt.plot(nearest_neighbors, training_accuracy, label = 'Training Accuracy')
plt.plot(nearest_neighbors, testing_accuracy, label = 'Testing Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Neighbors')
plt.legend()
#plt.show()

# Logistic Regression
logreg = LogisticRegression()
logreg = logreg.fit(X_train, y_train)

print('Logistic Training score: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Logistic Testing score: {:.2f}'.format(logreg.score(X_test, y_test)))

# Logistic Regression with high C
logreg_HC = LogisticRegression(C = 100)
logreg_HC = logreg_HC.fit(X_train, y_train)

print('Logistic Training score: {:.2f}'.format(logreg_HC.score(X_train, y_train)))
print('Logistic Testing score: {:.2f}'.format(logreg_HC.score(X_test, y_test)))

# Logistic Regression with low C
logreg_LC = LogisticRegression(C = 0.01)
logreg_LC = logreg_LC.fit(X_train, y_train)

print('Logistic Training score: {:.2f}'.format(logreg_LC.score(X_train, y_train)))
print('Logistic Testing score: {:.2f}'.format(logreg_LC.score(X_test, y_test)))

# Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=0)
tree = tree.fit(X_train, y_train)

print('Tree Training score: {:.2f}'.format(tree.score(X_train, y_train)))
print('Tree Testing score: {:.2f}'.format(tree.score(X_test, y_test)))

# Decision Tree Classifier with depth
tree_depth = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_depth = tree_depth.fit(X_train, y_train)

print('Tree with depth Training score: {:.3f}'.format(tree_depth.score(X_train, y_train)))
print('Tree with depth Testing score: {:.3f}'.format(tree_depth.score(X_test, y_test)))




