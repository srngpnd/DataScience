from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mglearn

# Original Dataset
boston = load_boston()
print('Shape: {}'.format(boston.data.shape))

# Dataset after feature engineering(combination of features)
X, y = mglearn.datasets.load_extended_boston()
print('Shape: {}'.format(X.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression()
lr = lr.fit(X_train, y_train)

print('Linear Training score: {}'.format(lr.score(X_train, y_train)))
print('Linear Testing score: {}'.format(lr.score(X_test, y_test)))

# Without Alpha
ridge = Ridge()
ridge = ridge.fit(X_train, y_train)

print('Ridge Training score: {}'.format(ridge.score(X_train, y_train)))
print('Ridge Testing score: {}'.format(ridge.score(X_test, y_test)))

# With Alpha setting HIgh alpha more restriction more generalization less training set performance
ridge_alpha = Ridge(alpha=10)
ridge_alpha = ridge_alpha.fit(X_train, y_train)

print('Ridge Alpha Training score: {}'.format(ridge_alpha.score(X_train, y_train)))
print('Ridge Alpha Testing score: {}'.format(ridge_alpha.score(X_test, y_test)))

# Verify ridge alpha constraint
plt.plot(ridge.coef_, 's', label = 'Ridge alpha = 1')
plt.plot(ridge_alpha.coef_, '^', label = 'Ridge alpha = 10')
plt.plot(lr.coef_, 'o', label = 'Linear Regression')
plt.xlabel('Coefficient Index')
plt.xlabel('Coefficient Magnitude')
plt.hlines(0,0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
#plt.show()

# Lasso
lasso = Lasso()
lasso = lasso.fit(X_train, y_train)

print('Lasso Training score: {:.2f}'.format(lasso.score(X_train, y_train)))
print('Lasso Testing score: {:.2f}'.format(lasso.score(X_test, y_test)))
print('NUmber of features used: {}'.format(np.sum(lasso.coef_ != 0)))

# Lasso with alpha
lasso_alpha = Lasso(alpha=0.01, max_iter=10000)
lasso_alpha = lasso_alpha.fit(X_train, y_train)

print('Lasso Training score: {:.2f}'.format(lasso_alpha.score(X_train, y_train)))
print('Lasso Testing score: {:.2f}'.format(lasso_alpha.score(X_test, y_test)))
print('NUmber of features used: {}'.format(np.sum(lasso_alpha.coef_ != 0)))

# Verify lasso alpha constraint
plt.plot(lasso.coef_, 's', label = 'Lasso alpha = 1')
plt.plot(lasso_alpha.coef_, '^', label = 'Lasso alpha = 0.01')
plt.plot(ridge.coef_, 'o', label = 'Ridge')
plt.xlabel('Coefficient Index')
plt.xlabel('Coefficient Magnitude')
plt.hlines(0,0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend(ncol = 2, loc = (0, 1.05))
plt.show()

