from .fedbase import BasicServer, BasicClient
from utils.fflow import Logger
import utils.fflow as flw
import os
import numpy as np
import torch

class MyLogger(Logger):
    def log(self, server=None, models=[]):
        if server == None or models == []: return
        if self.output == {}:
            self.output = {
                "meta": server.option,
                'test_metrics': [],
                'test_losses': [],
                'train_metrics': [],
                'train_losses': [],
                'valid_metrics': [],
                'valid_losses': [],
                'mean_test_metrics':[],
                'max_test_metrics':[]
            }

        for cid in range(server.num_clients):
            server.model = models[cid]
            test_metric, test_loss = server.test()
            train_metric, train_loss = server.clients[cid].test(server.model, 'train')
            # valid_metric, valid_loss = server.clients[cid].test(server.model, 'valid')
            self.output['test_metrics'].append(test_metric)
            self.output['test_losses'].append(test_loss)
            self.output['train_metrics'].append(train_metric)
            self.output['train_losses'].append(train_loss)
            # self.output['valid_metrics'].append(valid_metric)
            # self.output['valid_losses'].append(valid_loss)

        print("Training Loss:", self.output['train_losses'])
        print("Testing Loss:", self.output['test_losses'])
        # print("valid Accuracy:", self.output['valid_metrics'])
        print("Testing Accuracy:", self.output['test_metrics'])
        print("mean_Testing Accuracy:", np.mean(self.output['test_metrics']))
        print("max_Testing Accuracy:", np.max(self.output['test_metrics']))

#每次训练的train loss怎么看？


logger = MyLogger()

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None, validation=None):
        super(Server, self).__init__(option, model, clients, test_data, validation)

    def run(self):
        logger.time_start('Total Time Cost')
        selected_clients = [_ for _ in range(self.num_clients)]
        models = self.communicate(selected_clients)
        logger.time_end('Total Time Cost')
        logger.log(self, models)
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
        return

    def unpack(self, packages_received_from_clients):
        return [packages_received_from_clients[cid]['model'] for cid in range(self.num_clients)]

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        self.train(model)
        cpkg = self.pack(model)
        return cpkg

    def pack(self, model):
        return {
            "model": model,
        }

    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data)
                loss.backward()
                optimizer.step()
        return
