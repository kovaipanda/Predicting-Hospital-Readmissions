import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data8.csv"
data = pd.read_csv(file_path)

# Identify features and target
X = data.drop('Readmitted', axis=1)
y = data['Readmitted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

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
    early_stopping_rounds=10,  # Early stopping moved to the constructor
    **best_params
)

# Fit the classifier with early stopping
best_model = classifier.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)

# Predict and calculate metrics on the test set
y_pred_best = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

# Calculate evaluation metrics for the test set
accuracy_test = accuracy_score(y_test, y_pred_best)
precision_test = precision_score(y_test, y_pred_best, zero_division=1)
recall_test = recall_score(y_test, y_pred_best)
f1_test = f1_score(y_test, y_pred_best)
roc_auc_test = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# Calculate evaluation metrics for the training set
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train, zero_division=1)
recall_train = recall_score(y_train, y_pred_train)
f1_train = f1_score(y_train, y_pred_train)
roc_auc_train = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])

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

# Compare train vs test metrics to check for overfitting
print("\n--- Performance Comparison ---")
print(f'Test Set - Accuracy: {accuracy_test:.4f} (Train: {accuracy_train:.4f})')
print(f'Test Set - F1 Score: {f1_test:.4f} (Train: {f1_train:.4f})')

# Calculate and display the confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Readmitted", "Readmitted"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()


# Fine-Tuning: Focused Grid Search on reg_alpha and reg_lambda
param_grid_finetune = {
    'reg_alpha': 0, # 1
    'reg_lambda': 10 # 10
}

# Initialize a new XGBClassifier with the best parameters found earlier
classifier_finetune = XGBClassifier(
    eval_metric='logloss',
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    scale_pos_weight=best_params['scale_pos_weight'],
    min_child_weight=best_params['min_child_weight'],
    gamma=best_params['gamma'],
    reg_alpha=best_params['reg_alpha'],  # Starting point
    reg_lambda=best_params['reg_lambda']  # Starting point
)

# Create Stratified K-Folds object
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV for fine-tuning
grid_search_finetune = GridSearchCV(
    estimator=classifier_finetune,
    param_grid=param_grid_finetune,
    scoring='f1',
    cv=cv_strategy,
    verbose=1,
    n_jobs=-1
)
# Fit GridSearchCV for fine-tuning
grid_search_finetune.fit(X_train, y_train)

# Best parameters after fine-tuning
print("\n--- Fine-Tuning Best Parameters ---")
print("Best Parameters:", grid_search_finetune.best_params_)
print("Best Cross-Validation F1 Score:", grid_search_finetune.best_score_)

# Evaluate the fine-tuned model
fine_tuned_model = grid_search_finetune.best_estimator_

# Fit with early stopping
fine_tuned_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)

# Predictions
y_pred_finetuned = fine_tuned_model.predict(X_test)
y_pred_finetuned_train = fine_tuned_model.predict(X_train)

# Metrics for fine-tuned model on test set
accuracy_finetuned_test = accuracy_score(y_test, y_pred_finetuned)
precision_finetuned_test = precision_score(y_test, y_pred_finetuned, zero_division=1)
recall_finetuned_test = recall_score(y_test, y_pred_finetuned)
f1_finetuned_test = f1_score(y_test, y_pred_finetuned)
roc_auc_finetuned_test = roc_auc_score(y_test, fine_tuned_model.predict_proba(X_test)[:, 1])

# Metrics for fine-tuned model on training set
accuracy_finetuned_train = accuracy_score(y_train, y_pred_finetuned_train)
precision_finetuned_train = precision_score(y_train, y_pred_finetuned_train, zero_division=1)
recall_finetuned_train = recall_score(y_train, y_pred_finetuned_train)
f1_finetuned_train = f1_score(y_train, y_pred_finetuned_train)
roc_auc_finetuned_train = roc_auc_score(y_train, fine_tuned_model.predict_proba(X_train)[:, 1])

# Output fine-tuned results
print("\n--- Fine-Tuned Model Performance ---")
print("Test Set:")
print(f'Accuracy: {accuracy_finetuned_test:.4f}')
print(f'Precision: {precision_finetuned_test:.4f}')
print(f'Recall: {recall_finetuned_test:.4f}')
print(f'F1 Score: {f1_finetuned_test:.4f}')
print(f'ROC-AUC: {roc_auc_finetuned_test:.4f}')

print("\nTraining Set:")
print(f'Accuracy: {accuracy_finetuned_train:.4f}')
print(f'Precision: {precision_finetuned_train:.4f}')
print(f'Recall: {recall_finetuned_train:.4f}')
print(f'F1 Score: {f1_finetuned_train:.4f}')
print(f'ROC-AUC: {roc_auc_finetuned_train:.4f}')

# Calculate and display the confusion matrix for the fine-tuned model
cm_finetuned = confusion_matrix(y_test, y_pred_finetuned)
print("\nConfusion Matrix for Fine-Tuned Model:")
print(cm_finetuned)
disp_finetuned = ConfusionMatrixDisplay(confusion_matrix=cm_finetuned, display_labels=["Not Readmitted", "Readmitted"])
disp_finetuned.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for Fine-Tuned Model")
plt.show()