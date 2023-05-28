import numpy as np


def read_dataset(conf: dict):
    data_X, data_Y = [], []

    with open(conf['data_path']) as fin:
        for line in fin:
            data = line.split(',')
            data_X.append([float(e) for e in data[:-1]])
            if int(data[-1]) == 1:
                data_Y.append(1)
            else:
                data_Y.append(-1)

    data_X = np.array(data_X)
    data_Y = np.array(data_Y)

    idx = np.arange(data_X.shape[0])
    np.random.shuffle(idx)

    train_size = int(data_X.shape[0] * 0.8)

    train_x = data_X[idx[:train_size]]
    train_y = data_Y[idx[:train_size]]

    eval_x = data_X[idx[train_size:]]
    eval_y = data_Y[idx[train_size:]]

    return (train_x, train_y), (eval_x, eval_y)
