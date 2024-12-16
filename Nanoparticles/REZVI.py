import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read the file
df =pd.read_csv(r"D:\ALL CODES\REINCEFORCEMENT LEARNING\Papers\PAPERS\chemE RL papers\Nanoparticles\Silver nanomaterils antibacterial  .csv")
# Define columns exactly as specified
categorical_columns = [
    'Process', 'Steps', 'External_energy', 'NEW_Capping agent', 
    'NEW_capping agent_Class', 'NEW_reducing agent', 
    'NEW_reducing agent_class', 'Order of reagent_CODE', 
    'treatment', 'UVVIs PEAKS nm', 'shape', 
    'Method of determination_size', 'bacterial  Culture medium', 
    'bacterial  Species','Order of reagent_CODE'
]

numerical_columns = [
    'Temp_Celcius', 'Stirring', 'Synthesis_Duration_h', 
    'Scale_synthesis_ml', 'Precurcor_conc_mM', 
    'Capping agent concentration_mg/mL', 
    'Reducing agent quantity mL', 'core size (nm)', 
    'exposure dose concentration mg/mL', 'Exposure duration  (h)' 
      # This one is a numerical field despite being named as "code"
]

target_column = 'Bacteria reduction mm'

# Combine all columns we'll use
all_columns = categorical_columns + numerical_columns + [target_column]

# Select only the columns we're interested in
df_cleaned = df[all_columns].copy()

# Replace specific problematic values with 0 and 'stirring' with 1
df_cleaned.replace({'?': 0, 'no': 0, '': 0, 'stirring': 1, np.nan: 0}, inplace=True)

# Verify the changes
print(df_cleaned.head())




# Verify the change
print(df_cleaned['Stirring'].head())


# Verify the change
print(df_cleaned['Stirring'].head())


# Verify the change
print(df_cleaned['Stirring'].head())


# Preprocess categorical columns: Replace missing values with 'Unknown'
for col in categorical_columns:
    df_cleaned[col] = df_cleaned[col].astype('string').fillna('Unknown')

# Preprocess numerical columns: Convert to numeric, ensuring that any non-numeric values become NaN
# Inspect and Convert Column Types
for col in numerical_columns:
    if col in df_cleaned.columns:
        print(f"Column: {col}")
        print(f"Sample values before conversion:\n{df_cleaned[col].head()}\n")

        # Convert to numeric
        try:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            print(f"Sample values after conversion:\n{df_cleaned[col].head()}\n")
        except Exception as e:
            print(f"Error converting column {col}: {e}\n")
    else:
        print(f"Column {col} is missing in the dataframe.\n")




# Handle missing values in numerical columns by replacing NaN with column means
for col in numerical_columns:
    if col in df_cleaned.columns:
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

print("Final DataFrame:")
print(df_cleaned.info())



# Handle missing values in numerical columns by replacing NaN with column means
df_cleaned[numerical_columns] = df_cleaned[numerical_columns].fillna(df_cleaned[numerical_columns].mean())

# Remove rows with NaN in target column (Bacteria reduction mm)
df_cleaned.dropna(subset=[target_column], inplace=True)

# Apply OneHotEncoding to categorical columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
categorical_encoded = categorical_transformer.fit_transform(df_cleaned[categorical_columns])

# Prepare the data for imputation: Separate numerical columns
numerical_data = df_cleaned[numerical_columns]

# Check the shape of numerical data before imputation
print("Shape of numerical data before imputation:", numerical_data.shape)


# Impute missing values in numerical columns using IterativeImputer
imputer = IterativeImputer(
    estimator=GradientBoostingRegressor(random_state=42),
    max_iter=10,
    random_state=42
)

# Perform imputation
numerical_imputed = imputer.fit_transform(numerical_data)

# Check the shape of the imputed numerical data
print("Shape of numerical data after imputation:", numerical_imputed.shape)

# Reconstruct the dataframe with imputed numerical columns
df_imputed = df_cleaned.copy()

# Check if the number of columns in the imputed data matches the original DataFrame
print(f"Expected number of columns: {len(numerical_columns)}")
print(f"Number of columns in imputed data: {numerical_imputed.shape[1]}")

# If the number of columns matches, assign the imputed values to the DataFrame
if numerical_imputed.shape[1] == len(numerical_columns):
    df_imputed[numerical_columns] = numerical_imputed
else:
    print(f"Column mismatch: Expected {len(numerical_columns)} columns, but got {numerical_imputed.shape[1]}")

# Visualize the imputation results for numerical columns
def plot_imputation_results(original, imputed, column_names):
    plt.figure(figsize=(15, 10))
    
    # Create a subplot for each column
    for i, col in enumerate(column_names, 1):
        plt.subplot(4, 3, i)
        
        # Find indices of missing values
        missing_mask = np.isnan(original[:, i-1])
        
        # Convert original and imputed to 1D arrays for plotting
        original_values = original[:, i-1]
        imputed_values = imputed[: , i-1]
        
        # Plot original vs imputed for missing values
        plt.scatter(
            original_values[~missing_mask], 
            original_values[~missing_mask], 
            label='Original', 
            alpha=0.5
        )
        plt.scatter(
            original_values[missing_mask], 
            imputed_values[missing_mask], 
            label='Imputed', 
            color='red', 
            alpha=0.5
        )
        
        plt.title(col)
        plt.xlabel('Original Values')
        plt.ylabel('Imputed Values')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualize imputation results
plot_imputation_results(
    numerical_data.values, 
    numerical_imputed, 
    numerical_columns
)

# Print out number of missing values before and after imputation
print("Missing values before imputation:")
print(df_cleaned[numerical_columns].isnull().sum())

print("\nMissing values after imputation:")
print(df_imputed[numerical_columns].isnull().sum())

