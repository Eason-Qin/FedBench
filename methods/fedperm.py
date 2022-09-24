'''
Code credit to https://github.com/QinbinLi/MOON
for thier implementation of FedProx.
'''

from mimetypes import knownfiles
import torch
import logging
from methods.base import Base_Client, Base_Server
import copy
from torch.multiprocessing import current_process
from numpy.linalg import svd
import numpy as np
from gitrebasin.utils.weight_matching import weight_matching, apply_permutation, wideresnet_permutation_spec
from gitrebasin.utils.utils import  lerp

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.merge_lambda=client_dict['merge_lambda']
        self.model = self.model_type(22, 2, 0, num_classes=10).to(self.device)
        # print(self.in_channels)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        if len(server_state_dict)!=0:
            self.recover_model(server_state_dict)
    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights
    

    def recover_model(self,global_model):
        # print(Singular)
        permutation_spec = wideresnet_permutation_spec()
        final_permutation = weight_matching(permutation_spec,
                                        self.model.cpu().state_dict(), global_model)
        updated_params = apply_permutation(permutation_spec, final_permutation, global_model)
        model_a_dict = copy.deepcopy(self.model.cpu().state_dict())
        model_b_dict = copy.deepcopy(updated_params)
        permed_local = lerp(self.merge_lambda, model_a_dict, model_b_dict)
        self.model.load_state_dict(permed_local)
            

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(22, 2, 0, num_classes=10)
        print(self.in_channels)