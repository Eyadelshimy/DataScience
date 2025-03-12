import pandas as pd

# Load the dataset
df = pd.read_csv("./Datasets/loan_approval_dataset.csv")

# Display the first and last 12 rows
print(df.head(12))
print(df.tail(12))

# print the total number of rows and columns
rows, columns = df.shape
print(f"Total number of rows: {rows}")
print(f"Total number of columns: {columns}")

# list all column names along with their corresponding data types
print(df.dtypes)

# print the name of the first column
print(df.columns[0])

# Generate a summary of the dataset, including non-null counts and data types
print(df.info())

# Choose a categorical attribute and display the distinct values it contains
categorical_attribute = "Loan_Purpose"
distinct_values = df[categorical_attribute].unique()
print(f"Distinct values in '{categorical_attribute}': {distinct_values}")

# Identify the most frequently occurring value in the chosen categorical attribute "Loan_Purpose"
most_frequent_value = df["Loan_Purpose"].mode()[0]
print(
    f"The most frequently occurring value in 'Loan_Purpose' is: {most_frequent_value}"
)

# Calculate and present the mean, median, standard deviation, and percentiles for a numerical column
numerical_column = "Loan_Amount"

mean_value = df[numerical_column].mean()
median_value = df[numerical_column].median()
std_deviation = df[numerical_column].std()
percentile_20 = df[numerical_column].quantile(0.20)
percentile_50 = df[numerical_column].quantile(0.50)
percentile_80 = df[numerical_column].quantile(0.80)

print(f"Mean of '{numerical_column}': {mean_value}")
print(f"Median of '{numerical_column}': {median_value}")
print(f"Standard Deviation of '{numerical_column}': {std_deviation}")
print(f"20th Percentile of '{numerical_column}': {percentile_20}")
print(f"50th Percentile of '{numerical_column}': {percentile_50}")
print(f"80th Percentile of '{numerical_column}': {percentile_80}")

