from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
# print(X_train.shape)
# print(X_test.shape)

# MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(C=100)
svm.fit(X_train_scaled, y_train)

print('SVM accuracy rate: {:.3f}'.format(svm.score(X_test_scaled, y_test)))

# StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(C=100)
svm.fit(X_train_scaled, y_train)

print('SVM accuracy rate: {:.3f}'.format(svm.score(X_test_scaled, y_test)))

# PCA
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print('Original shape:{}'.format(X_scaled.shape))
print('PCA shape:{}'.format(X_pca.shape))




