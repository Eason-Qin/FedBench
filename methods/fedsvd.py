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
import json
import sys
import torch.nn as nn
from .reformmodel import resolver
import torchvision.models as models

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        # self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels)
        self.model = models.resnet18(pretrained=True)
        self.model.fc=nn.Linear(512,self.num_classes)
        # a=torch.load('/home/listu/yiqin/FED/Noisy_FL/Network/pretrain/CE/None0.0/ResNet12_0.ckpt')
        # weights_dict={}
        # for k, v in a.items():
            # print(k)
            # if 'module.' in k:
            #   new_k = k.replace('module.', '')
            #   weights_dict[new_k] = v
            # else: continue
        # print(weights_dict.keys())
        # del weights_dict['linear.weight']
        # del weights_dict['linear.bias']
        # del weights_dict['conv1.weight']
        # self.model.load_state_dict(weights_dict,strict=False)

        self.model=resolver(self.model).to(self.device)
        for name, param in self.model.named_parameters():
            param.requires_grad=False
            # print(param.requires_grad)
            if 'vector_S' in name or 'fc' in name:
               param.requires_grad = True
        # print(self.in_channels)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.U={}
        self.S={}
        self.VT={}
    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        if len(server_state_dict)!=0:
            self.recover_s(server_state_dict)
    def train(self):
        # train the local model
        self.model.to(self.device)
        for name, param in self.model.named_parameters():
            param.requires_grad=False
            # print(param.requires_grad)
            if 'vector_S' in name or 'fc' in name:
               param.requires_grad = True
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                # print(log_probs)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        self.extract_s()
        return self.model.state_dict()
    
    def extract_s(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                self.S[name]=param

    def recover_s(self,global_s):
        local_state_dict=self.model.state_dict()
        local_state_dict.update(global_s)
        self.model.load_state_dict(local_state_dict)

            

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = models.resnet18(pretrained=True)
        self.model.fc=nn.Linear(512,self.num_classes)
        # self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels)
        # a=torch.load('/home/listu/yiqin/FED/Noisy_FL/Network/pretrain/CE/None0.0/ResNet12_0.ckpt')
        # weights_dict={}
        # for k, v in a.items():
            # if 'module.' in k:
            #   new_k = k.replace('module.', '')
            #   weights_dict[new_k] = v
            # else: continue
        # del weights_dict['linear.weight']
        # del weights_dict['linear.bias']
        # self.model.load_state_dict(weights_dict,strict=False)
        self.model=resolver(self.model).to(self.device)
        for name, param in self.model.named_parameters():
            param.requires_grad=False
            print(param.requires_grad)
            if 'vector_S' in name or 'fc' in name:
               param.requires_grad = True
        self.global_model={}
        self.S_g={}
        self.U_g={}
        self.VT_g={}
    
    def start(self):
        with open('{}/config.json'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [{} for x in range(self.args.thread_number)]

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        '''clients aggregate weights'''
        cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]
        for name,param in self.model.named_parameters():
            # if param.requires_grad:
            self.global_model[name]=param
        for key in self.global_model:
            print(key)
            self.global_model[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)]).detach()
        self.recover_s(self.global_model)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        # print(self.global_model)
        return [self.global_model for x in range(self.args.thread_number)]
    
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
    
    def recover_s(self,global_s):
        local_state_dict=self.model.state_dict()
        local_state_dict.update(global_s)
        self.model.load_state_dict(local_state_dict)