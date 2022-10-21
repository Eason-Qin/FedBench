import torch
from methods.base import Base_Client, Base_Server
import torchvision.models as models
import torch.nn as nn
import logging

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        # self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels).to(self.device)
        self.model = models.resnet18(pretrained=True)
        # print(self.model.named_parameters)
        self.model.fc=nn.Linear(512,self.num_classes)
        self.prev_conv1_weight = self.model.conv1.weight.clone()
        self.now_conv1_weight = self.model.conv1.weight.clone()
        self.prev_conv2_weight = self.model.state_dict()['layer1.0.conv2.weight'].clone()
        self.now_conv2_weight = self.model.state_dict()['layer1.0.conv2.weight'].clone()
        self.prev_conv3_weight = self.model.state_dict()['layer2.0.conv2.weight'].clone()
        self.now_conv3_weight = self.model.state_dict()['layer2.0.conv2.weight'].clone()
        self.prev_conv4_weight = self.model.state_dict()['layer3.0.conv1.weight'].clone()
        self.now_conv4_weight = self.model.state_dict()['layer3.0.conv1.weight'].clone()
        self.prev_conv5_weight = self.model.state_dict()['fc.weight'].clone()
        self.now_conv5_weight = self.model.state_dict()['fc.weight'].clone()
        # print(self.in_channels)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.cos=torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def train(self):
        # train the local model
        
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        # print(self.model.conv1.weight.clone())
        self.prev_conv1_weight = self.model.conv1.weight.clone().view(1,-1)
        self.prev_conv2_weight = self.model.state_dict()['layer1.0.conv2.weight'].clone().view(1,-1)
        self.prev_conv3_weight = self.model.state_dict()['layer2.0.conv2.weight'].clone().view(1,-1)
        self.prev_conv4_weight = self.model.state_dict()['layer3.0.conv1.weight'].clone().view(1,-1)
        self.prev_conv5_weight = self.model.state_dict()['fc.weight'].clone().view(1,-1)
        print('cosof1beforeupdate',self.cos(self.prev_conv1_weight,self.now_conv1_weight))
        print('cosof2beforeupdate',self.cos(self.prev_conv2_weight,self.now_conv2_weight))
        print('cosof3beforeupdate',self.cos(self.prev_conv3_weight,self.now_conv3_weight))
        print('cosof4beforeupdate',self.cos(self.prev_conv4_weight,self.now_conv4_weight))
        print('cosof5beforeupdate',self.cos(self.prev_conv5_weight,self.now_conv5_weight))
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
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  '.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss)))
        self.now_conv1_weight = self.model.conv1.weight.clone().view(1,-1)
        self.now_conv2_weight = self.model.state_dict()['layer1.0.conv2.weight'].clone().view(1,-1)
        self.now_conv3_weight = self.model.state_dict()['layer2.0.conv2.weight'].clone().view(1,-1)
        self.now_conv4_weight = self.model.state_dict()['layer3.0.conv1.weight'].clone().view(1,-1)
        self.now_conv5_weight = self.model.state_dict()['fc.weight'].clone().view(1,-1)
        # print(self.now_conv1_weight.shape)
        print('cosof1',self.cos(self.prev_conv1_weight,self.now_conv1_weight))
        print('cosof2',self.cos(self.prev_conv2_weight,self.now_conv2_weight))
        print('cosof3',self.cos(self.prev_conv3_weight,self.now_conv3_weight))
        print('cosof4',self.cos(self.prev_conv4_weight,self.now_conv4_weight))
        print('cosof5',self.cos(self.prev_conv5_weight,self.now_conv5_weight))
        weights = self.model.cpu().state_dict()
        return weights
    
        
class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        # self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels)
        self.model = models.resnet18(pretrained=True)
        self.model.fc=nn.Linear(512,self.num_classes)
        print(self.in_channels)