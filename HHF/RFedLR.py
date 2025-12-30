import os
import csv
import os.path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
sys.path.append("..")
from Dataset.utils import init_logs, partition_data, get_dataloader, generate_proxy_data_indexs, mkdirs, get_proxy_dataloader
from myoptim import SelectiveBackPropSGD
from Network.Models_Def.lora import LoRA_ViT_timm
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import torch.backends.cudnn
import timm
from collections import OrderedDict
import torch

#20clients
'''
Global Parameters
'''
Seed = 0
TrainBatchSize = 256
TestBatchSize = 512
Pretrain_Epoch = 40
CommunicationEpoch = 40
Pariticpant_Params = {
    'loss_funnction' : 'CE',
    'optimizer_name' : 'SGD',
    'learning_rate'  : 0.01   #vit0.001 lora0.01
}
"""Noise Setting"""
Noise_type = 'symmetric' #['pairflip','symmetric',None]
Noise_rate = 0.6
"""Model Setting"""
N_Participants = 5
"""Private Dataset Setting"""
Private_Dataset_Name = 'cifar100' #['cifar10']
Private_Dataset_Dir = '../Dataset/cifar_100'
Private_Output_Channel = 100
Data_Partition = 'noniid' #['iid', 'noniid']
Noniid_Dirichlet_Beta = 0.5
"""Proxy Dataset Setting"""
Proxy_Dataset_Name = 'cifar100'
Proxy_Dataset_Dir = '../Dataset/cifar_100'
Proxy_Dataset_Length = 256
Proxy_Noise_Type = 'symmetric'
Robust_Ratio = 0.2 #0.1
"""Training method"""
Training_Method = 'lora_altfedavg'
"""Model Save Setting"""
Model_Save_Dir = '../Model_Storage/' + Private_Dataset_Name + '_' + Data_Partition + str(Noniid_Dirichlet_Beta) +'_' + str(Noise_type) + str(Noise_rate) + '/' + Training_Method
"""Reweight Para"""
Importance_Weight = 0.4

def count_trainable_parameters(model, robust_mask=None):
    """Count the number of trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    if robust_mask is None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad and p in robust_mask and robust_mask[p].any())
    return trainable_params, total_params

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB across all used GPUs"""
    if torch.cuda.is_available():
        total_memory = 0
        for device_id in [0, 1, 2, 3]: 
            try:
                with torch.cuda.device(device_id):
                    total_memory += torch.cuda.max_memory_allocated() / 1024**2
            except RuntimeError:
                continue
        return total_memory
    return 0

def evaluate_network(network,dataloader,logger):
    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        logger.info('Test Accuracy of the model on the test images: {} %'.format(acc))
    return acc

def train_one_epoch(model, dataloader):
    gradient_dict = dict()
    fisher_dict = dict()
    model.train()
    criterion = nn.CrossEntropyLoss()
    gradient_dict_A = dict()
    gradient_dict_B = dict()
    fisher_dict_A = dict()
    fisher_dict_B = dict()
    # Now begin
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'linear_a' in name:
                gradient_dict_A[param] = param.grad.abs() 
                fisher_dict_A[param] = (param.grad ** 2)
            elif 'linear_b' in name:
                gradient_dict_B[param] = param.grad.abs() 
                fisher_dict_B[param] = (param.grad ** 2)
            gradient_dict[param] = param.grad.abs() 
            fisher_dict[param] = (param.grad ** 2)
    return gradient_dict, fisher_dict, gradient_dict_A, gradient_dict_B, fisher_dict_A, fisher_dict_B

def flatten_gradient(gradient_dict):
    r = None
    for k, v in gradient_dict.items():
        v = v.view(-1).cpu().numpy()
        if r is None:
            r = v
        else:
           r = np.append(r, v)
    return r

def calculate_robust(clean_dataloader, noisy_dataloader, base_model, keep_ratio):
    # Training on clean and noisy data
    clean_updates, clean_fisher, clean_updates_A, clean_updates_B, clean_fisher_A, clean_fisher_B = train_one_epoch(base_model, clean_dataloader)
    noisy_updates, _, noisy_updates_A, noisy_updates_B, _, _ = train_one_epoch(base_model, noisy_dataloader)
    sensitivity = dict()
    importance = dict()
    importance_A = dict()
    importance_B = dict()
    for key in clean_updates:
        sensitivity[key] = min_max_normalize(clean_updates[key] - noisy_updates[key])
    for key in clean_fisher_A:
        importance_A[key] = min_max_normalize(clean_fisher[key] / (clean_updates[key] - noisy_updates[key] + 1e-8))
    for key in clean_fisher_B:
        importance_B[key] = min_max_normalize(clean_fisher[key] / (clean_updates[key] - noisy_updates[key] + 1e-8))
    r = flatten_gradient(sensitivity)
    polar = np.percentile(r, (1-keep_ratio)*100)
    robust_mask = dict()
    for k in sensitivity:
        robust_mask[k] = sensitivity[k] >= polar
    print('Robust Polar => {}'.format(polar))
    ra = flatten_gradient(importance_A)
    rb = flatten_gradient(importance_B)
    importance_mean = {'A':ra.mean(),'B':rb.mean()}
    if ra.mean() >= rb.mean(): #adaalt4 5 6
        freeze_matrix = 'B'
    else:
        freeze_matrix = 'A'
    print('ra/rb=>{}'.format(ra.mean()/(rb.mean()+1e-8)))
    return robust_mask, sensitivity, freeze_matrix, importance_mean

def update_model_via_private_data(device,network,private_dataloader,loss_function,optimizer_method,learning_rate,logger,robust_mask):
    if loss_function =='CE':
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    
    # Calculate initial parameter counts and GPU memory
    trainable_params, total_params = count_trainable_parameters(network, robust_mask)
    initial_memory = get_gpu_memory_usage()
    logger.info(f'Initial trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)')
    logger.info(f'Initial GPU memory usage: {initial_memory:.2f} MB')
    
    if robust_mask == None:
        if optimizer_method =='Adam':
            optimizer = optim.Adam(network.parameters(),lr=learning_rate)
        if optimizer_method =='SGD':
            optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        selback_optimizer = SelectiveBackPropSGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        selback_optimizer.set_gradient_mask(robust_mask)
    participant_local_loss_batch_list = []
    for batch_idx, (images, labels) in enumerate(private_dataloader):
        #--------------------------------
        images = images.to(device)
        labels = labels.to(device)
        outputs = network(images)
        labels = labels.long()
        loss = criterion(outputs, labels)
        #---------------------------------
        if robust_mask == None: 
            #---------------Original code------------------
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            #---------------Original code------------------
        else:
            selback_optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            selback_optimizer.step()
        logger.info('Private Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(images), len(private_dataloader.dataset),
            100. * batch_idx / len(private_dataloader), loss.item()))
    
    # Calculate final parameter counts and GPU memory
    final_memory = get_gpu_memory_usage()
    logger.info(f'Final GPU memory usage: {final_memory:.2f} MB')
    logger.info(f'Peak GPU memory usage: {final_memory - initial_memory:.2f} MB')
    
    return network, participant_local_loss_batch_list, robust_mask

def min_max_normalize(data):
    min_val = data.min()
    max_val = data.max()
    result = (data - min_val) / (max_val - min_val + 1e-8)
    return result

def robust_normalize(data):
    q1 = torch.quantile(data, 0.25)
    q3 = torch.quantile(data, 0.75)
    iqr = q3 - q1
    return (data - q1) / (iqr + 1e-8)

if __name__ =='__main__':
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    seed = Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_ids = [0,1,2,3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Reset GPU memory tracking for all devices
        for device_id in device_ids:
            try:
                with torch.cuda.device(device_id):
                    torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                continue
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info("Initialize Participants' Data idxs")
    _, net_dataidx_map = partition_data(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,partition=Data_Partition,
                                     num_classes=Private_Output_Channel,num_users=N_Participants,dirichlet_beta=Noniid_Dirichlet_Beta)
    net_datanum_map = {}
    data_total_num = 0
    for key, value in net_dataidx_map.items():
        net_datanum_map[key] = len(value)
        data_total_num += len(value)
    logger.info(net_datanum_map)

    logger.info("Load Participants' Models")
    all_models = []
    for i in range(N_Participants):
        network = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=Private_Output_Channel, pretrained_cfg_overlay=dict(file="../Network/Models_Def/vit_base_patch16_224.bin"))
        network = LoRA_ViT_timm(vit_model=network, r=4, alpha=4, num_classes=Private_Output_Channel)
        all_models.append(network)
    averaged_weights = OrderedDict()
    for i in range(N_Participants):
        local_weights = all_models[i].state_dict()
        for key in all_models[0].state_dict().keys():
            if i == 0:
                averaged_weights[key] = net_datanum_map[i] / data_total_num * local_weights[key]
            else:
                averaged_weights[key] += net_datanum_map[i] / data_total_num * local_weights[key]
                # print(local_weights[key])
                # print(averaged_weights[key])
    for i in range(N_Participants):
        all_models[i].load_state_dict(averaged_weights)

    logger.info("Initialize Proxy Data Parameters")
    proxy_data_indexs = generate_proxy_data_indexs(dataset=Proxy_Dataset_Name,datadir=Proxy_Dataset_Dir,size=Proxy_Dataset_Length)
    proxy_clean_train_dl, _, proxy_clean_train_ds, _ = get_proxy_dataloader(dataset=Proxy_Dataset_Name,datadir=Proxy_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                                                            dataidxs=proxy_data_indexs, noise_type=None, noise_rate=0)
    proxy_noisy_train_dl, _, proxy_noisy_train_ds, _ = get_proxy_dataloader(dataset=Proxy_Dataset_Name,datadir=Proxy_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                                                            dataidxs=proxy_data_indexs, noise_type=Proxy_Noise_Type, noise_rate=1.0)
    
    col_loss_list = []
    local_loss_list = []
    local_jsd_loss_list = []
    acc_list = []

    training_loss_file = os.path.join(Model_Save_Dir, 'training_loss.csv')
    test_accuracy_file = os.path.join(Model_Save_Dir, 'test_accuracy.csv')
    
    epoch_training_losses = []  
    epoch_test_accuracies = []  

    para_sensitivity = []
    for epoch_index in range(CommunicationEpoch):
        logger.info("The "+str(epoch_index)+" th Communication Epoch")

        '''
        Calculate Robust Mask and Freeze Matrix
        '''
        logger.info('Calculate Robust Mask and Freeze Matrix')
        participant_robust_mask_list = []
        freeze_matrix_cal_list = []
        importance_mean_list = []
        for participant_index in range(N_Participants):
            network = all_models[participant_index].to(device)
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            robust_mask, sensitivity, freeze_matrix_cal, _ = calculate_robust(proxy_clean_train_dl, proxy_noisy_train_dl, network, Robust_Ratio)
            participant_robust_mask_list.append(robust_mask)
            freeze_matrix_cal_list.append(freeze_matrix_cal)
        if freeze_matrix_cal_list.count('A') >= N_Participants/2:
            freeze_matrix = 'A'
            train_matrix = 'B'
        else:
            freeze_matrix = 'B'
            train_matrix = 'A'
        print('Freeze Matrix => {}'.format(freeze_matrix))
        
        '''
        Update Participants' Models via Private Data
        '''
        logger.info('Train Local Models')
        local_loss_batch_list = []
        para_sensitivity_list = []
        importance_mean_sum = 0.0
        for participant_index in range(N_Participants):
            network = all_models[participant_index].to(device)
            private_dataidx = net_dataidx_map[participant_index]
            train_dl, _, train_ds, _= get_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                                                    dataidxs=net_dataidx_map[participant_index], noise_type=Noise_type, noise_rate=Noise_rate)
            """Matrix select"""
            if freeze_matrix == 'A':
                network.freeze_lora_parameters(freeze_a=True, freeze_b=False)
                # learning_rate = Pariticpant_Params['learning_rate']
            else:
                network.freeze_lora_parameters(freeze_a=False, freeze_b=True)
                # learning_rate = Pariticpant_Params['learning_rate'] * 0.1
            
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()
            # robust_mask = None
            network,private_loss_batch_list, participant_robust_mask = update_model_via_private_data(device=device,network=network,private_dataloader=train_dl,
                                                                            loss_function=Pariticpant_Params['loss_funnction'],
                                                                            optimizer_method=Pariticpant_Params['optimizer_name'],
                                                                            learning_rate=Pariticpant_Params['learning_rate'],logger=logger, 
                                                                            robust_mask=None)
            network.module.freeze_lora_parameters(freeze_a=False, freeze_b=False)
            mean_private_loss_batch = np.mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_private_loss_batch)
            para_sensitivity_list.append(para_sensitivity)
            _, _, _, importance_mean = calculate_robust(proxy_clean_train_dl, proxy_noisy_train_dl, network, Robust_Ratio)
            importance_mean_list.append(importance_mean)
            importance_mean_sum += importance_mean[train_matrix]
            
            all_models[participant_index].load_state_dict(network.module.state_dict())
        local_loss_list.append(local_loss_batch_list)

        epoch_avg_loss = sum(local_loss_batch_list) / len(local_loss_batch_list)
        epoch_training_losses.append(epoch_avg_loss)
        logger.info(f'Epoch {epoch_index} average training loss: {epoch_avg_loss:.6f}')

        logger.info('Evaluate Models in the '+str(epoch_index)+'th Communication Epoch')
        acc_epoch_list = []
        for participant_index in range(N_Participants):
            private_dataset_dir = Private_Dataset_Dir
            _, test_dl, _, _= get_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize)
            network = all_models[participant_index].to(device)
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
            acc_epoch_list.append(accuracy)
        acc_list.append(acc_epoch_list)
        accuracy_avg = sum(acc_epoch_list) / N_Participants
        logger.info('Average Test Accuracy of the models on the test images: {} %'.format(accuracy_avg))

        epoch_test_accuracies.append(accuracy_avg)

        '''
        Model Aggregation (FedAvg)
        '''
        logger.info('Model Aggregation (FedAvg)')
        '''
        FedAvg_Importance-Reweight
        '''
        averaged_weights = OrderedDict()
        for i in range(N_Participants):
            local_weights = all_models[i].state_dict()
            for key in all_models[0].state_dict().keys():
                index_weight = Importance_Weight * importance_mean_list[i][train_matrix] / (importance_mean_sum + 1e-8) \
                                + (1-Importance_Weight)* net_datanum_map[i] / data_total_num
                # index_weight = index_weight_list[i]/index_weight_sum #test
                if i == 0:
                    averaged_weights[key] = index_weight * local_weights[key]
                else:
                    averaged_weights[key] += index_weight * local_weights[key]
                    # print(local_weights[key])
                    # print(averaged_weights[key])
        for i in range(N_Participants):
            all_models[i].load_state_dict(averaged_weights)

        """
        Evaluate Models in the final round
        """
        if epoch_index == CommunicationEpoch - 1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants): 
                _, test_dl, _, _= get_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize)
                network = all_models[participant_index].to(device)
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
                acc_epoch_list.append(accuracy)
            accuracy_avg = sum(acc_epoch_list) / N_Participants
            logger.info('Average Test Accuracy of the models on the test images: {} %'.format(accuracy_avg))

            # Log final GPU memory usage across all GPUs
            if torch.cuda.is_available():
                total_memory = 0
                for device_id in [0, 1, 2, 3]:
                    try:
                        with torch.cuda.device(device_id):
                            total_memory += torch.cuda.max_memory_allocated() / 1024**2
                    except RuntimeError:
                        continue
                logger.info(f'Total GPU memory usage across all GPUs: {total_memory:.2f} MB')

            logger.info('Save Models')
            mkdirs(Model_Save_Dir)
            for participant_index in range(N_Participants):
                network = all_models[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                torch.save(network.state_dict(), Model_Save_Dir + '/' + 'model_'+str(participant_index)+'.ckpt')

        """
        save log
        """
        logger.info('Saving training records')
        mkdirs(Model_Save_Dir)
        
        with open(training_loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Average Training Loss'])
            for epoch, loss in enumerate(epoch_training_losses):
                writer.writerow([epoch, loss])
        
        with open(test_accuracy_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Average Test Accuracy'])
            for epoch, acc in enumerate(epoch_test_accuracies):
                writer.writerow([epoch, acc])
        
        logger.info(f'Training records saved to {Model_Save_Dir}')
