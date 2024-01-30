import os
import pandas as pd

# Assuming you have a DataFrame named 'df' with a column 'filename'
# and 'directory_path' is the path where the files are expected to exist
directory_path = "Human Action Recognition/train"

# Example DataFrame
df = pd.read_csv("Human Action Recognition/train.csv")

# Function to check if a file exists in the specified directory
def file_exists(file_name):
    file_path = os.path.join(directory_path, file_name)
    return os.path.exists(file_path)

# Update the DataFrame in-place
df['file_exists'] = df['filename'].apply(file_exists)
df.drop(df[df['file_exists'] == False].index, inplace=True)
df.drop(columns=['file_exists'], inplace=True)

# Print the updated DataFrame
print("Updated DataFrame:")
df.to_csv('Human Action Recognition/data.csv')
