import argparse
import random
from server import *
from client import *
from utils.early_stopping import *
import datasets
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset('../data/', conf['type'])

    server = Server(conf, eval_datasets)
    clients = []

    for c in range(conf['no_models']):
        clients.append(Client(conf, train_datasets, c))

    early_stop = EarlyStopping(conf['checkpoint_path'])

    print('\n\n')
    for e in range(conf['global_epochs']):
        candidates = random.sample(clients, conf['k'])
        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            diff = c.local_train(server.global_model)

            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)

        acc, loss = server.model_eval()

        print(f'Round: {e}, acc: {acc:.3f}%, loss: {loss:.3f}')

        early_stop(loss, server.global_model)
