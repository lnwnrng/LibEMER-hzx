import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from tqdm import tqdm
from itertools import cycle

import argparse
import os.path
import time
from pathlib import Path

from utils.metric import Metric, SubMetric
from utils.store import make_output_dir
def save_state(output_dir, model, optimizer1,optimizer2,optimizer3, epoch, r_idx='last', rr_idx='last', metric=None, state='best'):
    # compatibility
    if type(output_dir) is argparse.Namespace:
        output_dir = make_output_dir(output_dir, output_dir.model)
    else:
        output_dir = Path(output_dir)
    if not ( r_idx == 'last' and rr_idx == 'last'):
        output_dir = output_dir / str(r_idx)
        output_dir = output_dir / str(rr_idx)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"An error occurred: {e.strerror}")
    checkpoint_path = output_dir / f'checkpoint-{str(epoch)}' if metric is None \
        else output_dir / f'checkpoint-{state}{metric}'
    save = {
        'model': model.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'optimizer3': optimizer3.state_dict(),
        'epoch': epoch,
    }
    torch.save(save, checkpoint_path)
    print(f"save model to {checkpoint_path}")
def train(model, dataset_train, dataset_val, dataset_test, device, output_dir='result/', metrics= None, metric_choose = None, optimizer1 = None, optimizer2= None,
          optimizer3=None, scheduler=None,batch_size = 16, epochs=40, criterion = None, loss_func = None, loss_param= None,test_sub_label = None):
    if metrics is None:
        metrics=['acc']
    if metric_choose is None:
        metric_choose = metrics[0]

    loss_metics = ['loss']
    
    train_length = len(dataset_train)
    val_length = len(dataset_val)
    test_length = len(dataset_test)

    train_drop_flag = train_length % batch_size == 1
    val_drop_flag = val_length % batch_size == 1 
    test_drop_flag = test_length % batch_size == 1
    #data sampler for train, val, test
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)

    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4,drop_last=train_drop_flag)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4,drop_last=val_drop_flag)
    data_loader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4,drop_last=test_drop_flag)
    test_sub_label_loader = DataLoader(
        test_sub_label, sampler=sampler_test, batch_size=batch_size, num_workers=4, drop_last=test_drop_flag
    ) if test_sub_label is not None else None

    model = model.to(device)

    best_metric = {s:0. for s in metrics}

    for epoch in range(epochs):
        model.train()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        metric = Metric(metrics)
        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train),desc=f'Train Epoch{epoch+1}/{epochs}: lr {optimizer3.param_groups[0]["lr"]}' )
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) and module.num_features ==1:
                module.eval()
        for idx, (eeg_features, bio_features, labels) in train_bar:
            #load the samples into device
            eeg_features = eeg_features.to(device)
            bio_features = bio_features.to(device)
            labels = labels.to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            labels = torch.argmax(labels, dim=1)
            prediction, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor, alpha = model(eeg_features, bio_features)
            partial_h1 = torch.from_numpy(partial_h1).to(torch.float).to(device)
            partial_h2 = torch.from_numpy(partial_h2).to(torch.float).to(device)

            predict_loss = criterion(prediction, labels) + (0 if loss_func is None else loss_func(loss_param))
            
            metric.update(torch.argmax(prediction, dim=1), labels, criterion(prediction, labels).item())
            train_bar.set_postfix_str(f'Classification Loss:{predict_loss.item():.2f}, CCA Loss:{cca_loss.item():.2f}')
            
            output1.backward(-0.1*partial_h1, retain_graph=True)
            output2.backward(-0.1*partial_h2, retain_graph=True)
            predict_loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
        if scheduler is not None:
            scheduler.step()
        print('\033[32m train state:'+metric.value())
        metric_value = evaluate(model, data_loader_val, device, metrics, criterion, loss_func, loss_param)
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer1,optimizer2, optimizer3,epoch, metric=m, state='best')
        
        if metric_value[metric_choose] == 1:
            break
        
    model.load_state_dict(torch.load(f'{output_dir}/checkpoint-best{metric_choose}')['model'])
    if test_sub_label is not None:
        metric_value = sub_evaluate(model, data_loader_test, test_sub_label_loader, device, metrics, criterion, loss_func, loss_param)
    else:
        metric_value = evaluate(model, data_loader_test, device, metrics, criterion, loss_func, loss_param)
    for m in metrics:
        print(f'best_val_{m}: {best_metric[m]:.2f}')
        print(f'best_test_{m}: {metric_value[m]:.2f}')

    return metric_value



@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, loss_func, loss_param):
    model.eval()
    metric = Metric(metrics)
    eval_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Evaluating:')

    for idx, (eeg_features, bio_features, labels) in eval_bar:
        eeg_features = eeg_features.to(device)
        bio_features = bio_features.to(device)
        labels = labels.to(device)
        labels = torch.argmax(labels, dim=1)

        prediction, _ ,_,_,_,_,_,_= model(eeg_features, bio_features)
        loss = criterion(prediction, labels) + (0 if loss_func is None else loss_func(loss_param))
        metric.update(torch.argmax(prediction, dim=1), labels, loss.item())

    print('\033[34m eval state:' + metric.value())
    return metric.values

@torch.no_grad()
def sub_evaluate(model, data_loader, sub_labels, device, metrics, criterion, loss_func, loss_param):
    model.eval()
    metric = SubMetric(metrics)
    
    eval_bar = tqdm(enumerate(zip(data_loader, sub_labels)), total= len(data_loader), desc = 'Evaluating:')

    for idx,((eeg_features, bio_features, labels), sub_label) in eval_bar:
        eeg_features = eeg_features.to(device)
        bio_features = bio_features.to(device)
        labels = labels.to(device)
        labels = torch.argmax(labels, dim=1)

        prediction, _ ,_,_,_,_,_,_= model(eeg_features, bio_features)
        loss = criterion(prediction, labels) + (0 if loss_func is None else loss_func(loss_param))

        metric.update(torch.argmax(prediction, dim =1),labels, sub_label, loss.item())
    
    print('\033[34m eval state:' + metric.value())
    return metric.values
