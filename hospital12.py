import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load the data
file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data7.csv'
data = pd.read_csv(file_path)

# Define the target variable
target = 'Readmitted'


# Define features and target
X = data.drop(columns=[target])
y = data[target]

# Handle imbalance using SMOTE (Optional)
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print(classification_report(y_test, y_pred))

# Save the updated data with predictions
data['Predicted_Readmitted'] = model.predict(X)
save_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data8.csv'
data.to_csv(save_path, index=False)
