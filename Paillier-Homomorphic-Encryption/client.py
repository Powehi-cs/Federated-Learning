import models
import numpy as np


class Client(object):

    def __init__(self, conf, public_key, weights, data_x, data_y):

        self.conf = conf

        self.public_key = public_key

        self.local_model = models.LRModel(public_key=self.public_key, w=weights, encrypted=True)

        self.data_x = data_x

        self.data_y = data_y

    def local_train(self, weights):

        original_w = weights

        self.local_model.set_encrypt_weights(weights)

        neg_one = self.public_key.encrypt(-1)

        for e in range(self.conf["local_epochs"]):
            print("start epoch ", e)

            idx = np.arange(self.data_x.shape[0])
            batch_idx = np.random.choice(idx, self.conf['batch_size'], replace=False)

            x = self.data_x[batch_idx]
            x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            y = self.data_y[batch_idx].reshape((-1, 1))

            batch_encrypted_grad = x.transpose() * (
                    0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one)
            encrypted_grad = batch_encrypted_grad.sum(axis=1) / y.shape[0]

            for j in range(len(self.local_model.encrypt_weights)):
                self.local_model.encrypt_weights[j] -= self.conf["lr"] * encrypted_grad[j]

        diff = []
        for j in range(len(self.local_model.encrypt_weights)):
            diff.append(self.local_model.encrypt_weights[j] - original_w[j])

        return diff
