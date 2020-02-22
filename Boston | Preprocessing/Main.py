from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2).fit(X_train_scaled)

X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

ridge = Ridge().fit(X_train_scaled, y_train)
print('Ridge with scaled: {}'.format(ridge.score(X_test_scaled, y_test)))
ridge = Ridge().fit(X_train_poly, y_train)
print('Ridge with poly: {}'.format(ridge.score(X_test_poly, y_test)))

rfr = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print('Ridge with scaled: {}'.format(rfr.score(X_test_scaled, y_test)))
rfr = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print('Ridge with poly: {}'.format(rfr.score(X_test_poly, y_test)))
