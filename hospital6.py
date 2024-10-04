import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
input_file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data4.csv'

# Load the encoded file into a DataFrame
df = pd.read_csv(input_file_path)

# Ensure 'Readmitted' is treated as a categorical variable
df['Readmitted'] = df['Readmitted'].astype('category')

# Separate features and target variable
X = df.drop(columns=['Readmitted'])
y = df['Readmitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict the target variable
y_pred = model.predict(X_test)

# Generate a classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print("\nFeature Importances:")
print(importance_df)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Save the feature importances to a CSV file
importance_output_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\feature_importances.csv'
importance_df.to_csv(importance_output_path, index=False)
print(f"Feature importances saved to {importance_output_path}")
