import torch
from methods.base import Base_Client, Base_Server
import torchvision.models as models
import torch.nn as nn
class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        # self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels).to(self.device)
        self.model = models.resnet18(pretrained=True)
        self.model.fc=nn.Linear(512,self.num_classes)
        # print(self.in_channels)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        
class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        # self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels)
        self.model = models.resnet18(pretrained=True)
        self.model.fc=nn.Linear(512,self.num_classes)
        print(self.in_channels)