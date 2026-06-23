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

# PART 1: CLASSIFICATION (Iris Dataset)
print("PART 1 - CLASSIFICATION")
iris = load_iris()
X_cls = iris.data
y_cls = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y_cls)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_cls = knn.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred_cls)
precision = precision_score(y_test, y_pred_cls, average='macro')
recall = recall_score(y_test, y_pred_cls, average='macro')
f1 = f1_score(y_test, y_pred_cls, average='macro')

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_cls, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_cls)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Iris Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\nInterpretation (Classification):")
print("- Accuracy is very high because Iris is an easy dataset.")
print("- Precision & Recall are both excellent → very few false positives/negatives.")
print("- Most important metric here: Accuracy is sufficient, but in real life (e.g. disease detection), Recall might be more critical.")
print("- Confusion Matrix shows almost perfect classification.\n")
