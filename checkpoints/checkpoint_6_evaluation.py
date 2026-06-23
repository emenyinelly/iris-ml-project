# Checkpoint: Understanding Evaluation Metrics
# Classification + Regression + Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes

plt.style.use('seaborn-v0_8')

print("=== EVALUATION METRICS CHECKPOINT ===\n")

# PART 2: REGRESSION (California Housing)
print("PART 2 - REGRESSION")
housing = fetch_california_housing()
X_reg = housing.data
y_reg = housing.target

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, 
                                                    test_size=0.2, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_reg = lr.predict(X_test)

# Regression Metrics
mae = mean_absolute_error(y_test, y_pred_reg)
mse = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_reg)

print(f"Mean Absolute Error (MAE)  : {mae:.4f}")
print(f"Mean Squared Error (MSE)   : {mse:.4f}")
print(f"Root Mean Squared Error    : {rmse:.4f}")
print(f"R² Score                   : {r2:.4f}")

print("\nInterpretation (Regression):")
print("- MAE tells us the average prediction error is about ${:.0f}k.".format(mae*100000))
print("- RMSE is higher than MAE → some larger errors exist.")
print("- R² = {:.2f} means the model explains about {:.1f}% of the variance in house prices.".format(r2, r2*100))
print("- The model is decent but likely underfitting (linear model on complex data).")
print("  We could improve it with more features, polynomial terms, or better models like Random Forest.\n")

