from DataScience.DataAnalysis import df
from tabulate import tabulate

#Apply a filter to select rows based on a specific condition of your choice (e.g., select records where a value exceeds a certain threshold).
threshold = 2.7
filtered_df = df[df['Interest_Rate'] < threshold]
print(tabulate(filtered_df, headers='keys', tablefmt='psql'))