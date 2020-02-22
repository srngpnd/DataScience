from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# print(X_train.shape)
# print(X_test.shape)

# MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC()
svm.fit(X_train_scaled, y_train)

print('SVM accuracy rate: {:.3f}'.format(svm.score(X_test_scaled, y_test)))

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv = 5)
grid.fit(X_train_scaled, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best set score: {:.2f}".format(grid.score(X_test_scaled, y_test)))
print("Best parameters: ", grid.best_params_)

# # StandardScaler
# scaler = StandardScaler()
# scaler = scaler.fit(X_train)
#
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# svm = SVC(C=100)
# svm.fit(X_train_scaled, y_train)
#
# print('SVM accuracy rate: {:.3f}'.format(svm.score(X_test_scaled, y_test)))
#
# # PCA
# scaler.fit(cancer.data)
# X_scaled = scaler.transform(cancer.data)
# pca = PCA(n_components=2)
# pca.fit(X_scaled)
# X_pca = pca.transform(X_scaled)
# print('Original shape:{}'.format(X_scaled.shape))
# print('PCA shape:{}'.format(X_pca.shape))




