import torch.utils.data
import torchvision
import models


class Server:
    def __init__(self, conf: dict, eval_dataset: torchvision.datasets.VisionDataset):
        self.conf = conf
        self.global_model = models.get_model(self.conf['model_name'])
        self.eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=self.conf['batch_size'],
            shuffle=True
        )

    def model_aggregate(self, weight_accumulator: dict, cnt: dict):
        for name, data in self.global_model.state_dict().items():
            if cnt[name] > 0:  # model compression
                # update_per_layer = weight_accumulator[name] * self.conf['lambda']
                update_per_layer = weight_accumulator[name] * (1.0 / cnt[name])
                if self.conf['dp']:
                    if torch.cuda.is_available():
                        noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, self.conf['sigma'])
                    else:
                        noise = torch.FloatTensor(update_per_layer.shape).normal_(0, self.conf['sigma'])
                    update_per_layer.add_(noise)
                if data.type() != update_per_layer.type():
                    data.add_(update_per_layer.to(torch.int64))
                else:
                    data.add_(update_per_layer)

    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.data.max(1)[1]  # first 1 mean search by row, second 1 mean max value index
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l
