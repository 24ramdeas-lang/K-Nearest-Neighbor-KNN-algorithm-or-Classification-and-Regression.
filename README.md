import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
df =pd.read_csv("diabetes (6) - diabetes (6).csv")
df.head()
import os
files = [f for f in os.listdir('.') if f.endswith('.csv')]
print("Available CSV files:", files)
import pandas as pd

d4 = pd.read_csv("diabetes (6) - diabetes (6).csv")

X = d4.drop("Outcome", axis=1)
y = d4["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
comparison = pd.DataFrame({
    "Original Mean": X_train.mean(),
    "Original Std": X_train.std(),
    "Scaled Mean": X_train_scaled.mean(axis=0),
    "Scaled Std": X_train_scaled.std(axis=0)
}, index=X_train.columns)

print(comparison)
k=5
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)

y_pred = knn.predict(X_test_scaled)

print("First 20 Predictions:")
print(y_pred[:20])
y_prob = knn.predict_proba(X_test_scaled)

print("First 20 Probabilities:")
print(y_prob[:20])
accuracy = knn.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)
k_values = range(1,31)

train_accuracy = []
test_accuracy = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train_scaled, y_train)

    train_accuracy.append(model.score(X_train_scaled, y_train))
    test_accuracy.append(model.score(X_test_scaled, y_test))
plt.figure(figsize=(10, 6))

plt.plot(k_values, train_accuracy, marker='o', label='Train Accuracy')
plt.plot(k_values, test_accuracy, marker='s', label='Test Accuracy')

plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title("KNN Accuracy vs. K")
plt.legend()
plt.grid(True)
plt.show()
best_k_accuracy = k_values[np.argmax(test_accuracy)]
best_accuracy = max(test_accuracy)

print("Best k value:", best_k_accuracy)
print("Best test accuracy:", best_accuracy)
from sklearn.model_selection import StratifiedKFold, cross_val_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
k_range = range(1, 31)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]
print(f"Optimal K: {best_k} with Mean CV Accuracy: {max(cv_scores):.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range, cv_scores, marker='o', color='b', linewidth=2)
plt.axvline(best_k, color='r', linestyle='--', label=f'Optimal K = {best_k}')
plt.title('5-Fold Stratified CV Accuracy vs K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Mean CV Accuracy')
plt.grid(True)
plt.legend()
plt.show()
best_k_accuracy = k_values[np.argmax(test_accuracy)]
best_accuracy = max(test_accuracy)

print("Best k value:", best_k_accuracy)
print("Best test accuracy:", best_accuracy)
from sklearn.model_selection import cross_val_score
cv_scores = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
best_k_cv = np.argmax(cv_scores) + 1
print("Best K based on Cross Validation:", best_k_cv)
print("Best CV Accuracy:", max(cv_scores))
param_grid = {
"n_neighbors": list (range(3, 31, 2)),
"weights": ["uniform", "distance"],
"metric": ["euclidean", "manhattan", "minkowski"],
"p": [1, 2]
}
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator = KNeighborsClassifier(),
    param_grid = param_grid,
    cv = 5,
    scoring = "accuracy",
    verbose = 1
)
grid_search.fit(X_train_scaled, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

best_knn = grid_search.best_estimator_
test_acc = best_knn.score(X_test_scaled, y_test)
print(f"Test set Accuracy with Best Parameters: {test_acc:.4f}")
