from hashlib import new
from http import client
from itertools import tee
from re import S
from struct import pack
from telnetlib import SE
from tkinter import E, W
from utils import fmodule
from .fedbase import BasicServer, BasicClient
import copy
import math
from utils.fmodule import add_gradient_updates, flatten, \
            mask_grad_update_by_order,add_update_to_model,compute_grad_update, unflatten,proportion_grad_update_by_order
import torch
import torch.nn.functional as F
import numpy as np
from torch.linalg import norm
from multiprocessing import Pool as ThreadPool
import scipy.stats
from main import logger
import utils.fflow as flw
import os

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None, validation =None):
        super(Server, self).__init__(option, model, clients, test_data, validation)
        
        self.reputation = [0 for i in range(len(self.clients))]
        self.alpha_reputation = option['alpha_reputatio'] 
        self.model = [model for i in range(len(self.clients))]
        self.Gamma = option['Gamma']
        self.Beta = option['Beta']
        self.contributions = []
        self.vol_data_clients = [0 for i in range(len(self.clients))]
        

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        corrs_agg = {}
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')
            self.iterate(round)
            self.global_lr_scheduler(round)
            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self, round=round, corrs_agg=corrs_agg)

        print("=================End==================")
        logger.time_end('Total Time Cost')
        # save results as .json file
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
        return
    
    def iterate(self, t):

        self.selected_clients = [i for i in range(self.num_clients)]
        models_before = copy.deepcopy(self.model)
        for i in range(self.num_clients):
            models_before[i].freeze_grad()

        loss_server = self.KL_test(self.model, 'test')
        loss_clients = self.KL_test(self.model, 'train')
        KL_clients = []
        KL = 0
        for i in range(len(self.clients)):
            len_data = len(loss_clients[i])
            len_server = len(loss_server[i])
            if len_data < len_server :
                loss_server[i] = loss_server[i][:len_data]
            else:
                loss_clients[i] = loss_clients[i][:len_server]
            KL = self.KL_divergence(loss_clients[i], loss_server[i])
            KL = 1/(KL)
            KL_clients.append(KL)
        KL_clients = torch.tensor(KL_clients)
        KL_clients = torch.div(KL_clients, sum(KL_clients))
        KL_clients = torch.div(KL_clients, torch.max(KL_clients))

        max_clients_data = max([len(self.clients[i].train_data) for i in range(self.num_clients)])
        for i in range(len(self.clients)):
            self.vol_data_clients[i] = torch.div(len(self.clients[i].train_data), max_clients_data)

        # training locally
        ws, losses = self.communicate(self.selected_clients)
        
        self.grads = []
        for i in range(len(self.clients)):
            gradient = compute_grad_update(old_model=self.model[i], new_model=ws[i])
            flattened = flatten(gradient)
            norm_value = norm(flattened) + 1e-7 
            gradient = unflatten(torch.multiply(torch.tensor(self.Gamma), torch.div(flattened,  norm_value)), gradient)
            self.grads.append(gradient)
        
        acc__, loss__ = self.validation_(ws)   
        weights = [0 for i in range(len(self.clients))]
        for i in range(len(self.clients)):
            weights[i] = acc__[i] * KL_clients[i] * KL_clients[i] 

        weight_ = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]

        # sum_weights = sum(weights)
        # for i in range(len(weights)):
        #     weights[i] = weights[i] / sum_weights
        # weights = acc__

        # 聚合梯度
        aggregated_gradient = [torch.zeros(param.shape) for param in self.model[0].parameters()]
        for gradient_, weight in zip(self.grads, weight_):
            aggregated_gradient = add_gradient_updates(aggregated_gradient, gradient_, weight=weight)

        # 对每个用户的声誉进行更新
        for i in range(self.num_clients):
            if t == 0:
                self.reputation[i] = weights[i]
            else:
                self.reputation[i] = self.alpha_reputation * self.reputation[i] + (1-self.alpha_reputation)*weights[i]
            self.reputation[i] = torch.tensor(self.reputation[i])
            self.reputation[i] = torch.clamp(self.reputation[i], min=1e-3)
        self.reputation = torch.tensor(self.reputation)
        self.reputation = torch.div(self.reputation, sum(self.reputation))

        q_ratios = torch.tanh(self.Beta * self.reputation)
        q_ratios = torch.div(q_ratios, torch.max(q_ratios))

        for i in range(len(self.clients)):
            q_ratios[i] = q_ratios[i] * KL_clients[i]

        self.contributions.append(q_ratios.numpy())

        self.reputation = self.reputation.tolist()
        if t >0 :
            for i in range(self.num_clients):
                reward_gradient = mask_grad_update_by_order(aggregated_gradient, mask_percentile=q_ratios[i], mode='layer')
                self.model[i] = copy.deepcopy(add_update_to_model(models_before[i], reward_gradient))
        return 


    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {"model" : copy.deepcopy(self.model[client_id])}
    
    def validation_(self, model_=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            the metric and loss of the model on the test data
        """
        if model_==None: model_=self.model
        eval_metrics, losses = [], [] 
        for i in range(self.num_clients):
            model = model_[i]

            if self.validation:
                model.eval()
                loss = 0
                eval_metric = 0
                data_loader = self.calculator.get_data_loader(self.validation, batch_size=64)
                for batch_id, batch_data in enumerate(data_loader):
                    bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
                    loss += bmean_loss * len(batch_data[1])
                    eval_metric += bmean_eval_metric * len(batch_data[1])
                eval_metric /= len(self.validation)
                loss /= len(self.validation)

                eval_metrics.append(eval_metric)
                losses.append(loss)
            else: return -1,-1
        return eval_metrics, losses


    def test(self, model_=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            the metric and loss of the model on the test data
        """
        if model_==None: model_=self.model
        eval_metrics, losses = [], [] 
        for i in range(self.num_clients):
            model = model_[i]

            if self.test_data:
                model.eval()
                loss = 0
                eval_metric = 0
                data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
                for batch_id, batch_data in enumerate(data_loader):
                    bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
                    loss += bmean_loss * len(batch_data[1])
                    eval_metric += bmean_eval_metric * len(batch_data[1])
                eval_metric /= len(self.test_data)
                loss /= len(self.test_data)

                eval_metrics.append(eval_metric)
                losses.append(loss)
            else: return -1,-1
        return eval_metrics, losses


    def test_on_clients(self, round, dataflag='valid'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            round: the current communication round
            dataflag: choose train data or valid data to evaluate
        :return
            evals: the evaluation metrics of the global model on each client's dataset
            loss: the loss of the global model on each client's dataset
        """
        evals, losses = [], []
        for c, model in zip(self.clients, self.model):
            eval_value, loss = c.test(model, dataflag)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses

    def KL_test(self, model_=None, data=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            the metric and loss of the model on the test data
        """
        if model_==None: model_=self.model
        eval_metrics, losses = [], [] 
        a = [i for i in range(len(self.clients))]
        for i,c in zip(a, self.clients):
            model = model_[i]

            if data == 'test':
                model.eval()
                loss = 0
                KL_loss = []
                eval_metric = 0
                data_loader = self.calculator.get_data_loader(self.validation, batch_size=64)
                for batch_id, batch_data in enumerate(data_loader):
                    bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
                    bmean_loss = [bmean_loss]
                    KL_loss.extend(bmean_loss)
                losses.append(KL_loss)

            elif data == 'train':
                loss = c.KL_test_(model, data)
                losses.append(loss)

        return losses
    
    # 计算两个分布的KL散度
    def KL_divergence(self, a, b):
        KL = scipy.stats.entropy(a,b)
        return KL

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        model.update_grad()
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

    def KL_test_(self, model, dataflag='valid'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            eval_metric: task specified evaluation metric
            loss: task specified loss
        """
        dataset = self.train_data if dataflag=='train' else self.valid_data
        model.eval()
        loss = 0
        eval_metric = 0
        losses = []
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
            bmean_loss = [bmean_loss]
            losses.extend(bmean_loss)
        return losses

