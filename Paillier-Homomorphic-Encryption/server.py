import models
import paillier
import numpy as np
import math


class Server(object):
    public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

    def __init__(self, conf, eval_dataset):

        self.conf = conf

        self.global_model = models.LRModel(public_key=Server.public_key, w_size=self.conf["feature_num"] + 1)

        self.eval_x = eval_dataset[0]

        self.eval_y = eval_dataset[1]

    def model_aggregate(self, weight_accumulator):

        for id, data in enumerate(self.global_model.encrypt_weights):
            update_per_layer = weight_accumulator[id] * self.conf["lambda"]

            self.global_model.encrypt_weights[id] = self.global_model.encrypt_weights[id] + update_per_layer

    def model_eval(self):
        correct = 0
        dataset_size = 0

        batch_num = math.ceil(self.eval_x.shape[0] / self.conf["batch_size"])

        self.global_model.weights = models.decrypt_vector(Server.private_key, self.global_model.encrypt_weights)

        for batch_id in range(batch_num):
            x = self.eval_x[batch_id * self.conf["batch_size"]: (batch_id + 1) * self.conf["batch_size"]]
            x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            y = self.eval_y[batch_id * self.conf["batch_size"]: (batch_id + 1) * self.conf["batch_size"]].reshape(
                (-1, 1))

            dataset_size += x.shape[0]

            wxs = x.dot(self.global_model.weights)

            pred_y = [1.0 / (1 + np.exp(-wx)) for wx in wxs]

            pred_y = np.array([1 if pred > 0.5 else -1 for pred in pred_y]).reshape((-1, 1))

            correct += np.sum(y == pred_y)

        acc = 100.0 * (float(correct) / float(dataset_size))

        return acc
