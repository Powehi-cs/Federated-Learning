import argparse, json
import random
from server import *
from client import *
import models
from datasets import read_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = read_dataset(conf)

    server = Server(conf, eval_datasets)
    clients = []

    train_size = train_datasets[0].shape[0]
    per_client_size = int(train_size / conf["no_models"])
    for c in range(conf["no_models"]):
        clients.append(Client(conf, Server.public_key, server.global_model.encrypt_weights,
                              train_datasets[0][c * per_client_size: (c + 1) * per_client_size],
                              train_datasets[1][c * per_client_size: (c + 1) * per_client_size]))

    for e in range(conf["global_epochs"]):

        server.global_model.encrypt_weights = models.encrypt_vector(Server.public_key,
                                                                    models.decrypt_vector(Server.private_key,
                                                                                          server.global_model.encrypt_weights))

        candidates = random.sample(clients, conf["k"])

        weight_accumulator = [Server.public_key.encrypt(0.0)] * (conf["feature_num"] + 1)

        for c in candidates:
            diff = c.local_train(server.global_model.encrypt_weights)

            for i in range(len(weight_accumulator)):
                weight_accumulator[i] = weight_accumulator[i] + diff[i]

        server.model_aggregate(weight_accumulator)

        acc = server.model_eval()

        print("Epoch %d, acc: %f\n" % (e, acc))
