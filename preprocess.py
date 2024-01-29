import pandas as pd


def shuffle_csv(pth, train_ratio, valid_ratio):
    """

    :param pth:
    :param train_ratio:
    :param valid_ratio:
    :return:
    """
    # Read the CSV file
    df = pd.read_csv(pth)

    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df[:int(len(df) * train_ratio )]
    valid_df = df[-int(len(df) * valid_ratio ):]

    return train_df, valid_df


if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_path = 'Human Action Recognition/Training_set.csv'

    # Call the function with the specified arguments
    train_set, valid_set = shuffle_csv(csv_path, .80, .20)

    # Print or use the resulting DataFrames as needed
    print("Training Set:")
    print(train_set.head())
    train_set.to_csv('Human Action Recognition/train.csv')

    print("\nValidation Set:")
    print(valid_set.head())
    train_set.to_csv('Human Action Recognition/test.csv')
