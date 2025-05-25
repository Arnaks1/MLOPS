import pandas as pd
import os


def download_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    os.makedirs('data/raw', exist_ok=True)

    df = pd.read_csv(url)
    df.to_csv('data/raw/titanic.csv', index=False)
    return df


if __name__ == "__main__":
    download_data()
