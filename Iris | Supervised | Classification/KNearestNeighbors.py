from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# import matplotlib.pyplot as plt
# from pandas.plotting import scatter_matrix
# import mglearn

iris_dataset = load_iris()

# print('Keys of iris dataset:/n{}'.format(iris_dataset.keys()))
# print('Shape of dataset:{}'.format(iris_dataset['data'].shape))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# print('X_train shape:{}'.format(X_train.shape))
# print('y_train shape:{}'.format(y_train.shape))
#
# print('X_test shape:{}'.format(X_test.shape))
# print('y_test shape:{}'.format(y_test.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])
# grr = scatter_matrix(iris_dataframe, c = y_train, figsize=(15, 15), marker='o', hist_kwds ={'bins':20}, s = 60, alpha = 0.8, cmap = mglearn.cm3)
# plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print('Score:{:.2f}'.format(knn.score(X_test, y_test)))