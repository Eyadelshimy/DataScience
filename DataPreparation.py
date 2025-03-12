from DataAnalysis import df
from tabulate import tabulate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Apply a filter to select rows based on a specific condition of your choice (e.g., select records where a value exceeds a certain threshold).
threshold = 2.7
filtered_df = df[df["Interest_Rate"] < threshold]
print(tabulate(filtered_df, headers="keys", tablefmt="psql"))

# Identify records where a chosen attribute starts with a specific letter and count how many records match this condition (Loan_Purpose starts with 'H')
count = df[df["Loan_Purpose"].str.startswith("H", na=False)].shape[0]
print(f"Number of records where Loan Purpose starts with 'H': {count}")

# Determine the total number of duplicate rows and remove them if found
count_dups = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Number of duplicate rows removed: {count_dups}")

# Convert the data type of a numerical column from integer to string(Loan_Amount)
df["Loan_Amount"] = df["Loan_Amount"].astype(str)

# Group the dataset based on two selected categorical features and analyze the results(Loan_Purpose and Loan_Term)
print(df.groupby(["Loan_Purpose", "Loan_Term"]).size())

# Check for the existence of missing values within the dataset
print(df.isnull().sum())

# If any missing values are found, replace them with the median or mode as appropriate
df.fillna(
    {col: df[col].median() for col in df.select_dtypes(include=["number"]).columns},
    inplace=True,
)
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Divide a chosen numerical column into 5 equal-width bins and count the number of records in each bin(Loan_Amount)
df["Loan_Amount"] = pd.to_numeric(df["Loan_Amount"])
df["Loan_Amount_Binned"] = pd.cut(df["Loan_Amount"], bins=5)
print(df["Loan_Amount_Binned"].value_counts())

# Identify and print the row corresponding to the maximum value of a selected numerical feature(Loan_Amount)
print(df[df["Loan_Amount"] == df["Loan_Amount"].max()])

# Construct a boxplot for an attribute you consider significant and justify the selection(Loan_Amount)(Justification: it is useful because it helps us see the distribution of loan amounts, including any outliers, as well as it shows the minimum, maximum, median, and quartiles, which help us understand how loan amounts vary)
plt.figure(figsize=(8, 6))
sns.boxplot(y=df["Loan_Amount"])
plt.title("Boxplot of Loan Amount")
plt.show()

# Generate a histogram for a chosen attribute and provide an explanation for its relevance(Loan_Amount)(Justification: Histogram is used to see how loan amounts are distributed across different ranges. It also helps to identify whether most loans fall within a certain amount and if the data is skewed)
plt.figure(figsize=(8, 6))
plt.hist(df["Loan_Amount"], bins=30, edgecolor="black")
plt.title("Histogram of Loan Amount")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()

# Create a scatterplot using two attributes and interpret the relationship observed(Loan_Amount vs Income)(Interpretation: This Scatterplot helps us check if there’s a relationship between how much a person earns and how much they borrow. If there's a pattern, it could suggest that higher incomes lead to higher loan amounts)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Loan_Amount"], y=df["Income"])
plt.title("Scatterplot of Loan Amount vs Income")
plt.xlabel("Loan Amount")
plt.ylabel("Applicant Income")
plt.show()

# Normalize the numerical attributes using StandardScaler to achieve standardized data
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=["number"]).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Perform PCA (Principal Component Analysis) to reduce dimensionality to two components, and visualize the dataset before and after applying PCA

# PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(df[numeric_columns])
df_pca = pd.DataFrame(data=pca_transformed, columns=["PC1", "PC2"])

# Scatterplot before PCA using first two numerical columns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df[numeric_columns[0]], y=df[numeric_columns[1]])
plt.title("Visualization Before PCA")
plt.xlabel(numeric_columns[0])
plt.ylabel(numeric_columns[1])
plt.show()

# Scatterplot after PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca["PC1"], y=df_pca["PC2"])
plt.title(" After PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Analyze the correlation between numerical features using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

