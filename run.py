import os
import json

with open('./config.json') as f:
    config=json.load(fp=f)

methods=['fedavg','fedsvd']
datasets=['data/emnist','data/fmnist','data/cifar100','data/cifar10']
client_num=[5,10,20]
epoch=[1,2,5,10]
com_round=[10,20,25,30]
alpha=[0.1,0.2,0.5]

for m in methods:
    for d in datasets:
        for c in client_num:
            for e in epoch:
                for cr in com_round:
                    for a in alpha:
                        os.system(f'python main.py --method {m} --data_dir {d} --public_data {d}\
                            --client_number {c} --thread_number {c} --comm_round {cr} --epochs {e}\
                            --partition_alpha {a}')

