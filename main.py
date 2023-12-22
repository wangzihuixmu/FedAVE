from builtins import print, sum
from torch.utils import tensorboard
import utils.fflow as flw
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr
import os
import csv


TensorWriter = SummaryWriter('./wzh/log/1001')
#  rffl 的计算

class MyLogger(flw.Logger):
    def log(self, server=None, round=None,corrs_agg=None):
        if server==None: return
        if self.output == {}:
            self.output = {
                "meta":server.option,
                "train_losses":[],
                "train_losses_clients":[],
                "test_accs":[],
                "test_losses":[],
                "valid_losses":[],
            }
        test_metric, test_loss = server.test()
        valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid')
        train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')
        self.output['train_losses'].append(1.0*sum([ck * closs for ck, closs in zip(server.client_vols, train_losses)])/server.data_vol)
        self.output['valid_losses'].append(1.0*sum([ck * closs for ck, closs in zip(server.client_vols, valid_losses)])/server.data_vol)
        self.output['train_losses_clients'].append(train_losses)

        self.output['test_accs'].append(test_metric)
        self.output['test_losses'].append(test_loss)
        print("Training Loss:", self.output['train_losses'][-1])
        print("valid Loss:", self.output['valid_losses'][-1])
        print("Testing Loss:", self.output['test_losses'][-1])
        print("Testing Accuracy:", self.output['test_accs'][-1])  
        print("Mean of testing Accuracy:", np.mean(self.output['test_accs'][-1]))
        print("Max of testing Accuracy:", np.max(self.output['test_accs'][-1]))

        # mnist-10 clients
        # powerlaw
        standalone_test_acc = [0.8076, 0.8863, 0.9026, 0.9114, 0.9362, 0.9345, 0.949, 0.9466, 0.9517, 0.9578]


        corrs = pearsonr(standalone_test_acc, self.output['test_accs'][-1])
        print("corrs:", corrs[0])

        TensorWriter.add_scalar('Training Loss', self.output['train_losses'][-1], round)
        TensorWriter.add_scalar('Mean of testing Accuracy', np.mean(self.output['test_accs'][-1]), round)
        TensorWriter.add_scalar('Max of testing Accuracy', np.max(self.output['test_accs'][-1]), round)
        TensorWriter.add_scalar('corrs', corrs[0], round)
        TensorWriter.add_scalar('valid_losses', self.output['valid_losses'][-1], round)

        corrs_agg[round] = corrs[0]
        corrs_agg = sorted(corrs_agg.items(), key = lambda kv:kv[1], reverse = True)
        if len(corrs_agg) >= 10 :
            max_corrs = corrs_agg[0:9:1]
            print("max_corrs:", max_corrs)

logger = MyLogger()

def main(): 
    # read options
    option = flw.read_option()
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main()


