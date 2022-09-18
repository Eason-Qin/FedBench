import torch
from methods.base import Base_Client, Base_Server
import torch.nn.functional as F
import numpy as np

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels).to(self.device)
        # print(self.in_channels)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            # self.load_client_state_dict(received_info)
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader)*self.args.batch_size
            weights = self.train()
            acc = self.test()
            client_results.append({'weights':weights, 'num_samples':num_samples,'acc':acc, 'client_index':self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results
    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            correct_count=[]
            incorrect_count=[]
            for batch_idx, (x, target) in enumerate(self.test_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                # loss = self.criterion(pred, target)
                pred = F.softmax(pred, 1)
                _, predicted = torch.max(pred, 1)
                # for p,g in zip(predicted.cpu().numpy(),target.cpu().numpy()):
                    # if p == g:
                        # correct_count.append(g)
                    # if p!=g:
                        # incorrect_count.append(g)
                correct = predicted.eq(target).sum()
                # print(predicted.eq(target))

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            # c_u,c_u_b=np.unique(correct_count,return_counts=True)
            # i_c_u,i_c_u_b=np.unique(incorrect_count,return_counts=True)
            # tmp = {c_u[i]: c_u_b[i] for i in range(len(c_u))}
            # print("************* Client {} Correct = {} **************".format(self.client_index, str(tmp)))
            # tmp = {i_c_u[i]: i_c_u_b[i] for i in range(len(i_c_u))}
            # print("************* Client {} InCorrect = {} **************".format(self.client_index, str(tmp)))
            acc = (test_correct / test_sample_number)*100
            print("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
        return acc
        
class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(class_num=self.num_classes,in_channels=self.in_channels)
        print(self.in_channels)
    
