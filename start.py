import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the file path from the environment variable
file_path = os.getenv('LOAN_APPROVAL_DATASET_PATH')

# Load the dataset
df = pd.read_csv(file_path)

# Display the first and last 12 rows
print(df.head(12))
print(df.tail(12))