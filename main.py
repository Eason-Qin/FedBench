'''
Main file to set up the FL system and train
Code design inspired by https://github.com/FedML-AI/FedML
'''
import torch
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import numpy as np
import random
import data_preprocessing.data_loader as dl
import argparse
from models.resnet import resnet56, resnet18
from models.resnet_gradaug import resnet56 as resnet56_gradaug
from models.resnet_gradaug import resnet18 as resnet18_gradaug
from models.resnet_stochdepth import resnet56 as resnet56_stochdepth
from models.resnet_stochdepth import resnet18 as resnet18_stochdepth
from models.resnet_fedalign import resnet56 as resnet56_fedalign
from models.resnet_fedalign import resnet18 as resnet18_fedalign
from models.resnet_hhf import ResNet12 as resnet12_hhf
from models.resnet_hhf import ResNet50 as resnet50_hhf
from models.resnet_fedsvg import ResNet12 as resnet12_svg
from models.mobilnet_v2 import MobileNetV2
from models.shufflenet import ShuffleNetG2
from gitrebasin.models.resnet import ResNet

from torch.multiprocessing import set_start_method, Queue, get_context
import multiprocessing
import logging
import os
from collections import defaultdict
import time
import json
from argparse import Namespace
import shutil

# methods
import methods.fedavg as fedavg
import methods.fedperm as fedperm
import methods.fedsvd as fedsvd
import methods.gradaug as gradaug
import methods.fedprox as fedprox
import methods.sino as sino
import methods.moon as moon
import methods.stochdepth as stochdepth
import methods.mixup as mixup
import methods.fedalign as fedalign
import methods.HHF as HHF
import methods.fedun as fedun
from methods.reformmodel import resolver
import data_preprocessing.custom_multiprocess as cm
from mail import send_email
# from SMS import send_message
def add_args(parser):
    # Training settings
    parser.add_argument('--method', type=str, default='fedavg', metavar='N',
                        help='Options are: fedavg, fedprox, moon, mixup, stochdepth, gradaug, fedalign')

    parser.add_argument('--data_dir', type=str, default='data/cifar10',
                        help='data directory: data/cifar100, data/cifar10, or another dataset')
    
    parser.add_argument('--public_data', type=str, default='data/cifar100',
                        help='data directory: data/cifar100, data/cifar10, or another dataset')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local clients')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--client_number', type=int, default=5, metavar='NN',
                        help='number of clients in the FL system')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=20, metavar='EP',
                        help='how many epochs will be trained locally per round')

    parser.add_argument('--comm_round', type=int, default=25,
                        help='how many rounds of communications are conducted')

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='test pretrained model')

    parser.add_argument('--mu', type=float, default=0.45, metavar='MU',
                        help='mu value for various methods')

    parser.add_argument('--width', type=float, default=0.25, metavar='WI',
                        help='minimum width for subnet training')

    parser.add_argument('--mult', type=float, default=1.0, metavar='MT',
                        help='multiplier for subnet training')

    parser.add_argument('--num_subnets', type=int, default=3,
                        help='how many subnets sampled during training')

    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')

    parser.add_argument('--thread_number', type=int, default=16, metavar='NN',
                        help='number of parallel training threads')

    parser.add_argument('--client_sample', type=float, default=1.0, metavar='MT',
                        help='Fraction of clients to sample')

    parser.add_argument('--stoch_depth', default=0.5, type=float,
                    help='stochastic depth probability')

    parser.add_argument('--gamma', default=0.0, type=float,
                    help='hyperparameter gamma for mixup')

    parser.add_argument('--gpus', default=[0], type=list,
                    help='GPUs can be used.')
    
    parser.add_argument('--merge_lambda', default=0.5, type=float,
                    help='model merge lambda.')

    args = parser.parse_args()

    return args

# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Helper Functions
def init_process(q, Client):
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])

def run_clients(received_info):
    try:
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info('exiting')
        return None

def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        if args.client_sample<1.0:
            num_clients = int(args.client_number*args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number==0 and num_clients>0:
            clients_per_thread = int(num_clients/args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t+clients_per_thread)]
                mapping_dict[c].append(idxs)
        else:
            raise ValueError("Sampled client number not divisible by number of threads")
    return mapping_dict

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed()
    # get arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args = vars(args)
    # with open("./config.json", mode="w") as f:
        # json.dump(args.__dict__, f, indent=4)
    with open('./config.json') as f:
        args.update(json.load(fp=f))
    args=Namespace(**args)
 
    # get data
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict,\
         class_num,y_train,y_test,net_idx_map = dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size,args.public_data)
    # dl.print_partition_data("datapart.png",args.client_number,class_num,y_train,y_test,net_idx_map)
    mapping_dict = allocate_clients_to_threads(args)
    gpus=args.gpus
    if 'mnist' in args.data_dir:
        in_channel=1
    else:
        in_channel=3
    #init method and model type
    if args.method=='fedavg':
        Server = fedavg.Server
        Client = fedavg.Client
        Model = resnet18 if 'cifar'or 'mnist' in args.data_dir else resnet18
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num,'in_channels':in_channel}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': gpus[i % len(gpus)],
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num,'in_channels':in_channel} for i in range(args.thread_number)]

    elif args.method=='gradaug':
        Server = gradaug.Server
        Client = gradaug.Client
        Model = resnet56_gradaug if 'cifar' in args.data_dir else resnet18_gradaug
        width_range = [args.width, 1.0]
        resolutions = [32, 28, 24, 20] if 'cifar' in args.data_dir else [224, 192, 160, 128]
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 
                            'width_range': width_range, 'resolutions': resolutions} for i in range(args.thread_number)]
    elif args.method=='fedprox':
        Server = fedprox.Server
        Client = fedprox.Client
        Model = resnet56 if 'cifar' in args.data_dir else resnet18
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]

    elif args.method=='sino':
        # specify client
        Server = sino.Server
        Client = sino.Client
        Model = resnet18 if 'cifar' or 'mnist' in args.data_dir else resnet18
        # args needed in Clients.init()
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num,'in_channels':in_channel}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': gpus[i % len(gpus)],
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num,'in_channels':in_channel} for i in range(args.thread_number)]
    
    elif args.method=='fedsvd':
        # specify client
        Server = fedsvd.Server
        Client = fedsvd.Client
        Model = resnet12_svg if 'cifar' or 'mnist' in args.data_dir else resnet12_svg
        # args needed in Clients.init()
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num,'in_channels':in_channel}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': gpus[i % len(gpus)],
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num,'in_channels':in_channel} for i in range(args.thread_number)]
    
    elif args.method=='permu':
        # specify client
        Server = fedperm.Server
        Client = fedperm.Client
        Model = ResNet
        # args needed in Clients.init()
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num,'in_channels':in_channel}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': gpus[i % len(gpus)],
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num,'in_channels':in_channel,'merge_lambda':args.merge_lambda} for i in range(args.thread_number)]

    

    elif args.method=='moon':
        Server = moon.Server
        Client = moon.Client
        Model = resnet56 if 'cifar' in args.data_dir else resnet18
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method=='stochdepth':
        Server = stochdepth.Server
        Client = stochdepth.Client
        Model = resnet56_stochdepth if 'cifar' in args.data_dir else resnet18_stochdepth
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method=='mixup':
        Server = mixup.Server
        Client = mixup.Client
        Model = resnet56 if 'cifar' in args.data_dir else resnet18
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method=='fedalign':
        Server = fedalign.Server
        Client = fedalign.Client
        Model = resnet56_fedalign if 'cifar' in args.data_dir else resnet18_fedalign
        width_range = [args.width, 1.0]
        resolutions = [32] if 'cifar' in args.data_dir else [224]
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 
                            'width_range': width_range, 'resolutions': resolutions} for i in range(args.thread_number)]
    else:
        raise ValueError('Invalid --method chosen! Please choose from availible methods.')
    
    #init nodes, put global and local args in queue
    client_info = Queue()
    for i in range(args.thread_number):
        client_info.put((client_dict[i], args))

    # Start server and get initial outputs
    # init clients with args
    pool = cm.MyPool(processes=args.thread_number, initializer=init_process, initargs=(client_info, Client))
    # init server
    server_dict['save_path'] = '{}/logs/{}__{}_e{}_c{}_{}'.format(os.getcwd(),
        time.strftime("%Y%m%d_%H%M%S"), args.method, args.epochs, args.client_number,args.data_dir.split('/')[-1])
    if not os.path.exists(server_dict['save_path']):
        os.makedirs(server_dict['save_path'])
    shutil.copy("./config.json",server_dict['save_path'])
    server = Server(server_dict, args)
    server_outputs = server.start()
    # Start Federated Training
    time.sleep(150*(args.client_number/16)) #  Allow time for threads to start up
    for r in range(args.comm_round):
        logging.info('************** Round: {} ***************'.format(r))
        round_start = time.time()
        client_outputs = pool.map(run_clients, server_outputs)
        client_outputs = [c for sublist in client_outputs for c in sublist]
        server_outputs = server.run(client_outputs)
        round_end = time.time()
        logging.info('Round {} Time: {}s'.format(r, round_end-round_start))
    pool.close()
    pool.join()
    send_message(f"{server_dict['save_path']}__training finished.")