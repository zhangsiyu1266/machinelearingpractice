import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data = load_breast_cancer()
X_all = data.data
y_all = data.target.astype(float)
X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
print("训练集形状:", X.shape)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
n_features = X.shape[1]
w = np.zeros(n_features)
b = 0.0
learning_rate = 0.1
for epoch in range(1000):
    z = X @ w + b
    p = sigmoid(z)
    gradient_w = X.T @ (p - y)/ len(y)
    gradient_b = np.mean(p-y)
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b
z_test = X_test @ w + b
p_test = sigmoid(z_test)
prediction_test = (p_test >= 0.5).astype(int)
accuracy_test = np.mean(prediction_test == y_test)
print("逻辑回归Accuracy =", accuracy_test)
print("w =", w)
print("b =", b)
X0 = X[y == 0]
X1 = X[y == 1]
mu0 = np.mean(X0, axis=0)
mu1 = np.mean(X1, axis=0)
Sw = (X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)
w_lda =  np.linalg.inv(Sw) @ (mu0 - mu1)
center0 = (w_lda @ mu0)
center1 = (w_lda @ mu1)
threshold = (w_lda @ mu0 + w_lda @ mu1) / 2
if center1  > center0:
    prediction_lda = (proj_test >= threshold) .astype(int)
else:
    prediction_lda = (proj_test < threshold) .astype(int)
accuracy_lda= np.mean(prediction_lda == y_test)
print("LDA Accuracy =", accuracy_lda)


