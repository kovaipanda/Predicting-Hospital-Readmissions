import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import pickle

# Load the dataset
file_path = r"C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data8.csv"
data = pd.read_csv(file_path)

# Identify features and target
X = data.drop('Readmitted', axis=1)
y = data['Readmitted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=100, stratify=y)

# Define the classifier with fixed hyperparameters from previous GridSearch
best_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.01,
    'subsample': 1.0,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1.9995049504950495,
    'min_child_weight': 2,
    'gamma': 0.2,
    'reg_alpha': 0,
    'reg_lambda': 0.01
}

classifier = XGBClassifier(
    eval_metric='logloss',
    early_stopping_rounds=10,
    **best_params
)

# Fit the classifier with early stopping
final_model = classifier.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)

# Save the trained model to a pickle file
model_file_path = r"C:\Users\DELL\Downloads\hospital_resubmisssion\xgb_model.pkl"
with open(model_file_path, 'wb') as model_file:
    pickle.dump(final_model, model_file)

print(f"Model saved to {model_file_path}")

# Load the model from the pickle file (for future use)
with open(model_file_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Predict and calculate metrics on the test set using the loaded model
y_pred_best = loaded_model.predict(X_test)
y_pred_train = loaded_model.predict(X_train)

# Calculate evaluation metrics for the test set
accuracy_test = accuracy_score(y_test, y_pred_best)
precision_test = precision_score(y_test, y_pred_best, zero_division=1)
recall_test = recall_score(y_test, y_pred_best)
f1_test = f1_score(y_test, y_pred_best)
roc_auc_test = roc_auc_score(y_test, loaded_model.predict_proba(X_test)[:, 1])

# Calculate evaluation metrics for the training set
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train, zero_division=1)
recall_train = recall_score(y_train, y_pred_train)
f1_train = f1_score(y_train, y_pred_train)
roc_auc_train = roc_auc_score(y_train, loaded_model.predict_proba(X_train)[:, 1])

# Output the results
print("\n--- Test Set Performance ---")
print(f'Accuracy: {accuracy_test:.4f}')
print(f'Precision: {precision_test:.4f}')
print(f'Recall: {recall_test:.4f}')
print(f'F1 Score: {f1_test:.4f}')
print(f'ROC-AUC: {roc_auc_test:.4f}')

print("\n--- Training Set Performance ---")
print(f'Accuracy: {accuracy_train:.4f}')
print(f'Precision: {precision_train:.4f}')
print(f'Recall: {recall_train:.4f}')
print(f'F1 Score: {f1_train:.4f}')
print(f'ROC-AUC: {roc_auc_train:.4f}')

# Calculate and display the confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Readmitted", "Readmitted"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

# Plot ROC curve for both training and test sets
plt.figure(figsize=(10, 6))

# ROC Curve for Test Set
fpr_test, tpr_test, _ = roc_curve(y_test, loaded_model.predict_proba(X_test)[:, 1])
roc_auc_test = auc(fpr_test, tpr_test)
plt.plot(fpr_test, tpr_test, color='blue', label=f'Test Set (AUC = {roc_auc_test:.2f})')

# ROC Curve for Training Set
fpr_train, tpr_train, _ = roc_curve(y_train, loaded_model.predict_proba(X_train)[:, 1])
roc_auc_train = auc(fpr_train, tpr_train)
plt.plot(fpr_train, tpr_train, color='green', label=f'Train Set (AUC = {roc_auc_train:.2f})')

# Plot random chance line
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

# Titles and labels
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

# Save performance metrics to CSV
metrics_test = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Value': [accuracy_test, precision_test, recall_test, f1_test, roc_auc_test]
}
df_metrics_test = pd.DataFrame(metrics_test)
df_metrics_test.to_csv(r"C:\Users\DELL\Downloads\hospital_resubmisssion\test_metrics.csv", index=False)

metrics_train = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Value': [accuracy_train, precision_train, recall_train, f1_train, roc_auc_train]
}
df_metrics_train = pd.DataFrame(metrics_train)
df_metrics_train.to_csv(r"C:\Users\DELL\Downloads\hospital_resubmisssion\train_metrics.csv", index=False)

# Save confusion matrix to CSV
cm_df = pd.DataFrame(cm, index=["Not Readmitted", "Readmitted"], columns=["Predicted Not Readmitted", "Predicted Readmitted"])
cm_df.to_csv(r"C:\Users\DELL\Downloads\hospital_resubmisssion\confusion_matrix.csv")
