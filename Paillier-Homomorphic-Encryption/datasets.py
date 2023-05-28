import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(conf: dict):
    df = pd.read_csv(conf['data_path'], header=None)
    label = df.iloc[:, -1].values
    df = df.iloc[:, :-1].values

    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=0.1)

    return (x_train, y_train), (x_test, y_test)
