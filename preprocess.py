import os
import pandas as pd

def shuffle_csv(pth, train_ratio, valid_ratio, data_dir):
    # Read the CSV file
    df = pd.read_csv(pth)

    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Add prefix to the "filename" column
    df['filename'] = 'Human Action Recognition/train/' + df['filename']

    # Remove filenames that don't exist in the directory
    df['file_exists'] = df['filename'].apply(lambda x: os.path.exists(x))
    df = df[df['file_exists']]

    # Drop the 'file_exists' column as it is no longer needed
    df = df.drop(columns=['file_exists'])

    # Split into training and validation sets
    train_df = df[:int(len(df) * train_ratio)]
    valid_df = df[-int(len(df) * valid_ratio):]

    return train_df, valid_df

if __name__ == "__main__":
    # Specify the path to your CSV file and data directory
    csv_path = 'Human Action Recognition/train.csv'
    data_directory = 'Human Action Recognition/train'

    # Call the function with the specified arguments
    train_set, valid_set = shuffle_csv(csv_path, 0.80, 0.20, data_directory)

    # Print or use the resulting DataFrames as needed
    print("Training Set:")
    print(train_set.head())
    train_set.to_csv('Human Action Recognition/train.csv', index=False)

    print("\nValidation Set:")
    print(valid_set.head())
    valid_set.to_csv('Human Action Recognition/test.csv', index=False)
