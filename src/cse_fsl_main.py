"""
Add local loss, no need to transmit the gradient
client transmit smashed data not every batch data
server part: model 0 batch 0 - model 0 batch 4 - model 1 batch 0 - model 1 batch 4  ...
"""

import os
import time
import math
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import options, utils
from train import model


print("Whether we are using GPU: ", torch.cuda.is_available())
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
    
class Client():
    def __init__(self, id, train_loader, c_args):
        if c_args['dataset'] == "cifar":
            self.model = model.Client_model_cifar() 
            self.auxiliary_model = model.Auxiliary_model_cifar()
        elif c_args['dataset'] == "femnist":
            self.model = model.Client_model_femnist() 
            self.auxiliary_model = model.Auxiliary_model_femnist()
        self.criterion = nn.NLLLoss().to(DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=c_args["lr"])       
        self.auxiliary_criterion = nn.NLLLoss().to(DEVICE)
        self.auxiliary_optimizer = optim.SGD(self.auxiliary_model.parameters(), lr=c_args["lr"])
        
        self.train_loader = train_loader
        self.epochs = c_args['epoch']
        self.dataset_size = len(self.train_loader) * c_args["batch_size"] 

class Server():
    def __init__(self, c_args):
        if c_args['dataset'] == "cifar":
            self.model = model.Server_model_cifar()
        elif c_args['dataset'] == "femnist":
            self.model = model.Server_model_femnist()
        self.criterion = nn.NLLLoss().to(DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=c_args["lr"])


def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)   

def calculate_load(model):        
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2    # MB
#     size_all_mb = (param_size + buffer_size)   # B
    return size_all_mb

if __name__ == '__main__':    
    ## get system configs
    args = options.args_parser('CSE_FSL')    #---------todo
    u_args, s_args, c_args = options.group_args(args) #---------todo
    utils.show_utils(u_args) #---------todo
    
    seed = u_args['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## process dataset
    trainSet, testSet = utils.get_dataset(s_args, u_args) 
    client_train_set, client_test_set = utils.depart_dataset(u_args, s_args, trainSet, testSet)

    
    trainLoader_list = []
    for i in range(s_args["activated"]):
        train_set = client_train_set[i]["idxs"]
        trainLoader_list.append(DataLoader(utils.DatasetSplit(trainSet, train_set), batch_size=c_args['batch_size'], shuffle=True, pin_memory=False))
    
    
    testLoader = DataLoader(testSet, batch_size=c_args['batch_size'], shuffle=False, pin_memory=False)
    
    
    # Define the server, and the list of client copies
    server = Server(c_args)
    client_copy_list = []
  
    for i in range(s_args["activated"]):   
        client_copy_list.append(Client(i, trainLoader_list[i], c_args))
      
    # Initial client model
    # Initial server model
    init_all(client_copy_list[0].model, torch.nn.init.normal_, mean=0., std=0.05) 
    init_all(client_copy_list[0].auxiliary_model, torch.nn.init.normal_, mean=0., std=0.05) 
    init_all(server.model, torch.nn.init.normal_, mean=0., std=0.05) 
    
        
    for i in range(s_args["activated"]):
        client_copy_list[i].model.load_state_dict(client_copy_list[0].model.state_dict())
        client_copy_list[i].model.to(DEVICE)
        client_copy_list[i].auxiliary_model.load_state_dict(client_copy_list[0].auxiliary_model.state_dict())
        client_copy_list[i].auxiliary_model.to(DEVICE)
    server.model.to(DEVICE)    
    
        
    # # Calculate the weights for dataset size
    dataset_size_list = [client_copy_list[i].dataset_size for i in range(s_args["activated"])]
    total = sum(dataset_size_list)
    factor = [i / total for i in dataset_size_list]
    print("Aggregation Factor: ", factor)


    r = 0  # current communication round
    acc_list = []
    loss_list = []
    comm_load_list = []
    start = time.time()
    comm_load = 0
    batch_max_round = total // c_args["batch_size"] // s_args["activated"]
    
    while r < s_args["round"]:
        it_list = []
        for i in range(s_args["activated"]):
            it_list.append(iter(trainLoader_list[i]))
        batch_round = u_args['batch_round']
        max_batch = batch_round
        client_i = 0
        start_index = 0
        while max_batch <= batch_max_round and client_i < s_args["activated"]: 
            cur_client_index = start_index
            while cur_client_index < batch_max_round and cur_client_index < max_batch:                   
                samples, labels = next(it_list[client_i])
                client_copy_list[client_i].optimizer.zero_grad()
                samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
                
                # client feedforward
                splitting_output = client_copy_list[client_i].model(samples)
                local_smashed_data = splitting_output.clone().detach().requires_grad_(True)
                smashed_data = splitting_output.clone().detach().requires_grad_(True)
                
                # client calculates the local loss and do the backpropagation and update auxiliary model weights
                client_copy_list[client_i].auxiliary_optimizer.zero_grad()
                local_output = client_copy_list[client_i].auxiliary_model(local_smashed_data)
                local_loss = client_copy_list[client_i].auxiliary_criterion(local_output, labels)
                local_loss.backward()  
                client_copy_list[client_i].auxiliary_optimizer.step()

                # client backpropagation and update client-side model weights
                gradient = local_smashed_data.grad
                splitting_output.backward(gradient)
                client_copy_list[client_i].optimizer.step()
         
                # server feedforward, calculate loss, backpropagation and update server-side model weights 
                if cur_client_index == max_batch - 1:
                    server.optimizer.zero_grad()
                    comm_load += smashed_data.numel() * 4   # float32 = 4 bytes
                    output = server.model(smashed_data)
                    loss = server.criterion(output, labels)
                    loss.backward()       
                    server.optimizer.step() 
                
                cur_client_index += 1
            
            if client_i == s_args["activated"] - 1:
                client_i = 0
                start_index += batch_round
                max_batch += batch_round
            else:
                client_i += 1
                    
                    

        # ===========================================================================================
        # Model Aggregation (weighted)
        # ===========================================================================================

        # Initial the aggregated model and its weights
        aggregated_client = copy.deepcopy(client_copy_list[0].model)
        aggregated_client_weights = aggregated_client.state_dict()
        aggregated_client_auxiliary = copy.deepcopy(client_copy_list[0].auxiliary_model)
        aggregated_client_weights_auxiliary = aggregated_client_auxiliary.state_dict()

        for key in aggregated_client_weights:
            aggregated_client_weights[key] = client_copy_list[0].model.state_dict()[key] * factor[0]
        for key in aggregated_client_weights_auxiliary:
            aggregated_client_weights_auxiliary[key] = client_copy_list[0].auxiliary_model.state_dict()[key] * factor[0]

        for i in range(1, s_args["activated"]):
            for key in aggregated_client_weights:
                aggregated_client_weights[key] += client_copy_list[i].model.state_dict()[key] * factor[i]
            for key in aggregated_client_weights_auxiliary:
                aggregated_client_weights_auxiliary[key] += client_copy_list[i].auxiliary_model.state_dict()[key] * factor[i]
    

        # Update client model weights and auxiliary weights
        for i in range(s_args["activated"]):
            client_copy_list[i].model.load_state_dict(aggregated_client_weights)
            client_copy_list[i].auxiliary_model.load_state_dict(aggregated_client_weights_auxiliary)
            comm_load += 2 * calculate_load(client_copy_list[i].model)
            comm_load += 2 * calculate_load(client_copy_list[i].auxiliary_model)

            
        # ===========================================================================================
        # Inference
        # ===========================================================================================
        aggregated_client.to(DEVICE)
        aggregated_client.load_state_dict(aggregated_client_weights)
        test_correct = 0
        test_loss = []
        for samples, labels in testLoader:
            samples, labels = samples.to(DEVICE).float(), labels.to(DEVICE).long()
            splitting_output = aggregated_client(samples)
            output = server.model(splitting_output)
            batch_loss = server.criterion(output, labels)
            test_loss.append(batch_loss.item())
            _, predicted = torch.max(output.data, 1)
            test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
        loss = sum(test_loss) / len(test_loss)
        print(
            '\nRound {}, for the weighted aggregated final model, testing loss: {:.2f}, testing acc: {:.2f}%  ({}/{})'
                .format(r, loss, 100. * test_correct / len(testLoader.dataset), test_correct, len(testLoader.dataset)))

        acc_list.append(test_correct / len(testLoader.dataset))
        loss_list.append(loss)
        comm_load_list.append(comm_load)
        r += 1

    print('The total running time for all rounds is ', round(time.time() - start, 2), 'seconds')
    print("Testing accuracy:", acc_list)
    print("Testing loss:", loss_list)
    
    '''
        Save reults to .json files.
    '''
    results = {'test_loss': loss_list, 'test_acc' : acc_list,
               'comm_load' : comm_load_list, 'step': s_args['t_round']}

    file_name = os.path.join(u_args['save_path'], 'results.json')
    with open(file_name, 'w') as outf:
        json.dump(results, outf)
        print(f"\033[1;36m[NOTICE] Saved results to '{file_name}'.\033[0m")
#     torch.save([acc_list, loss_list, comm_load_list], u_args['save_path'])