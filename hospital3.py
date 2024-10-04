import pandas as pd
from sklearn.preprocessing import LabelEncoder

# File paths
input_file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data.csv'
output_file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data2.csv'

# Load the file into a DataFrame
df = pd.read_csv(input_file_path)

# Initialize a LabelEncoder
label_encoders = {}
mapping_dict = {}

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Convert categorical columns to numerical using LabelEncoder
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    
    # Store the mapping of original values to encoded values
    mapping_dict[column] = dict(zip(le.classes_, le.transform(le.classes_)))
    label_encoders[column] = le

# Save the modified DataFrame to a new CSV file
df.to_csv(output_file_path, index=False)

# Display the mapping for each categorical column
print("Mappings of Categorical Columns:")
for column, mapping in mapping_dict.items():
    print(f"{column}: {mapping}")

print(f"\nCategorical columns converted to numerical and saved to {output_file_path}")
