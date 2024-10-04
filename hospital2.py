import pandas as pd
from scipy.stats import chi2_contingency

# File path
input_file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data.csv'

# Load the encoded file into a DataFrame
df = pd.read_csv(input_file_path)

# Ensure 'Readmitted' is treated as a categorical variable
df['Readmitted'] = df['Readmitted'].astype('category')

# Identify categorical features (excluding the target variable)
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features.remove('Readmitted')  # Remove target variable from the list

# Initialize a list to hold results
chi_square_results = []

# Perform Chi-square tests
for feature in categorical_features:
    # Create a contingency table
    contingency_table = pd.crosstab(df[feature], df['Readmitted'])
    
    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Append results
    chi_square_results.append({
        'Feature': feature,
        'Chi2 Statistic': chi2,
        'P-Value': p,
        'Degrees of Freedom': dof
    })

# Create a DataFrame from results
results_df = pd.DataFrame(chi_square_results)

# Display the results
print("Chi-square Test Results:")
print(results_df)

# Save results to a CSV file
results_output_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\chi_square_results.csv'
results_df.to_csv(results_output_path, index=False)
print(f"Chi-square results saved to {results_output_path}")
