import models, torch, copy
import torch.utils.data
import torchvision


class Client:
    def __init__(self, conf: dict, train_dataset: torchvision.datasets.VisionDataset, id=-1):
        self.conf = conf
        self.local_model = models.get_model(conf['model_name'])
        self.client_id = id
        self.train_dataset = train_dataset
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])  # train_data_size per client
        train_indices = all_range[id * data_len: (id + 1) * data_len]
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=conf['batch_size'],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        )

        # model sparsity
        self.mask = {}
        for name, param in self.local_model.state_dict().items():
            p = torch.ones_like(param) * self.conf['prop']
            if torch.is_floating_point(param):
                self.mask[name] = torch.bernoulli(p)
            else:
                self.mask[name] = torch.bernoulli(p).long()

    def local_train(self, model: torch.nn.Module):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])

        self.local_model.train()
        for e in range(self.conf['local_epochs']):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                if self.conf['dp']:
                    model_norm = models.model_norm

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = data - model.state_dict()[name]
            diff[name] = self.mask[name] * diff[name]  # element-wise multiply

        # model compression
        diff = sorted(diff.items(), key=lambda item: abs(torch.mean(item[1].float())), reverse=True)  # return list
        ret_size = int(self.conf['rate'] * len(diff))
        return dict(diff[:ret_size])

    def local_train_malicious(self, model: torch.nn.Module):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])

        pos = []
        for i in range(2, 28):
            pos.append([i, 3])
            pos.append([i, 4])
            pos.append([i, 5])

        self.local_model.train()
        for e in range(self.conf['local_epochs']):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                for k in range(self.conf["poisoning_per_batch"]):  # backdoor attack
                    img = data[k].numpy()
                    for i in range(0, len(pos)):
                        img[0][pos[i][0]][pos[i][1]] = 1.0
                        img[1][pos[i][0]][pos[i][1]] = 0
                        img[2][pos[i][0]][pos[i][1]] = 0
                    target[k] = self.conf['poison_label']

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

                optimizer.step()

                if self.conf['dp']:
                    model_norm = models.model_norm(model, self.local_model)
                    norm_scale = min(1, self.conf['C'] / model_norm)
                    for name, data in self.local_model.state_dict().items():
                        clipped_difference = norm_scale * (data - model.state_dict()[name])
                        data.copy_(model.state_dict()[name] + clipped_difference)

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = self.conf['eta'] * (data - model.state_dict()[name]) + model.state_dict()[name]

        return diff
