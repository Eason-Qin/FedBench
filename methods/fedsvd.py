'''
Code credit to https://github.com/QinbinLi/MOON
for thier implementation of FedProx.
'''

import torch
import logging
from methods.base import Base_Client, Base_Server
import copy
from torch.multiprocessing import current_process
from numpy.linalg import svd
import numpy as np
import json
import sys

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels).to(self.device)
        # print(self.in_channels)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.U={}
        self.S={}
        self.VT={}
    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        if len(server_state_dict)!=0:
            self.recover_model(server_state_dict)
    def train(self):
        # train the local model
        self.model.to(self.device)
        global_weight_collector = copy.deepcopy(list(self.model.parameters()))
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
        self.svd_local_model()
        # weights = self.model.cpu().state_dict()
        return self.S
    
    def svd_local_model(self):
        for name, param in self.model.named_parameters():
            if 'conv' in name:
                # print(name)     
                parameter=param.cpu().detach().numpy()
                parameter=parameter.reshape(param.size(0),-1)
                U_, S_, VT_ = svd(parameter,full_matrices=False)
                self.U[name]=U_
                self.S[name]=S_
                self.VT[name]=VT_

    def recover_model(self,Singular):
        # print(Singular)
        statedict=self.model.state_dict()
        for name in self.S:
            self.S[name]=Singular[name]
        for layer_name in self.S:
            if len(self.S[layer_name]) != min(self.U[layer_name].shape[0],self.VT[layer_name].shape[0]):
                # print("extending!")
                np.append(self.S[layer_name],np.array([0*(min(self.U[layer_name].shape[0],self.VT[layer_name].shape[0])-len(self.S[layer_name]))]))
            paramshape=statedict[layer_name].shape
            layer_weights=np.dot(np.dot(self.U[layer_name], np.diag(self.S[layer_name])),self.VT[layer_name])
            layer_weights=layer_weights.reshape(paramshape)
            statedict[layer_name]=torch.tensor(layer_weights).to(self.device)
            
        self.model.load_state_dict(statedict)
            

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels)
        self.S_g={}
    
    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [{} for x in range(self.args.thread_number)]

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        
        '''clients aggregate weights'''
        cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]
        for key in client_sd[0]:
            self.S_g[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.S_g for x in range(self.args.thread_number)]
    
    def log_info(self, client_info, acc):
        client_acc = sum([c['acc'] for c in client_info])/len(client_info)
        out_str = 'Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n'.format(acc, client_acc, self.round)
        client_indv_acc=[c['acc'] for c in client_info]
        client_str=f'Client acc : {client_indv_acc}\n'
        client_sd = [c['weights'] for c in client_info]
        dict_size=f'Client communictaion size : {round(sys.getsizeof(client_sd[0])/ 1024, 2)} KB\n'
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write(out_str)
            out_file.write(client_str)
            out_file.write(dict_size)
