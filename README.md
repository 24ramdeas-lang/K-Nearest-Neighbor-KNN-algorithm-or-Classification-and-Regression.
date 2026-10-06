Library Imports

pandas, numpy → For data handling and numerical operations.

matplotlib, seaborn → For visualization.

sklearn.model_selection → For splitting data and cross-validation.

sklearn.preprocessing.StandardScaler → For feature scaling.

sklearn.neighbors → For KNN classifier and regressor.

Dataset Loading

The code loads the Diabetes dataset (diabetes (6) - diabetes (6).csv) using pandas.read_csv().

It checks available .csv files in the directory to confirm dataset presence.

Features (X) are separated from the target variable (Outcome).

Train-Test Split

Data is split into training (80%) and testing (20%) sets using train_test_split().

Stratified sampling ensures balanced class distribution.

Feature Scaling

Standardization is applied using StandardScaler.

Comparison of original mean/std vs scaled mean/std is printed to show normalization effect.

Model Training (Basic KNN)

A KNN classifier with 
𝑘
=
5
 neighbors is trained on the scaled training data.

Predictions (y_pred) and probabilities (y_prob) are generated for the test set.

Accuracy score is printed.

Accuracy vs K Plot

The model is trained for different values of 
𝑘
 (1–30).

Training and testing accuracies are stored and plotted against 
𝑘
.

The best 
𝑘
 value and corresponding accuracy are identified.

Cross-Validation

Stratified K-Fold (5 folds) is applied to evaluate stability of results.

Mean accuracy for each 
𝑘
 is calculated.

Optimal 
𝑘
 is selected based on cross-validation scores.

A plot shows CV accuracy vs 
𝑘
.

Hyperparameter Tuning (Grid Search)

A parameter grid is defined for:

n_neighbors (3–30, odd values)

weights (uniform, distance)

metric (euclidean, manhattan, minkowski)

p (1, 2 for Minkowski distance)

GridSearchCV is used to find the best hyperparameters.

Best parameters and CV accuracy are printed.

Final model is tested on the test set with tuned parameters.
