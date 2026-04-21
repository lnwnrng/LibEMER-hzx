import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from tqdm import tqdm
from itertools import cycle

from utils.metric import Metric, SubMetric
from utils.store import save_state

def train(model, dataset_train, dataset_val, dataset_test, device, output_dir='result/', metrics= None, metric_choose = None, optimizer = None, scheduler=None,batch_size = 16, epochs=40, 
          class_criterion = None, corr_criterion = None, finegrained_criterion = None, coarsegrained_criterion = None,ce_criterion = None, lambda_c=0.5, lambda_v = 0.1,loss_func = None, loss_param= None,test_sub_label = None):
    if metrics is None:
        metrics=['acc']
    if metric_choose is None:
        metric_choose = metrics[0]

    loss_metics = ['loss']
    
    #data sampler for train, val, test
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)
    
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4)
    data_loader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4)
    test_sub_label_loader = DataLoader(
        test_sub_label, sampler=sampler_test, batch_size=batch_size, num_workers=4, drop_last=False
    ) if test_sub_label is not None else None

    model = model.to(device)

    best_metric = {s:0. for s in metrics}
    value_previous = 0
    count = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        metric = Metric(metrics)
        
        train_bar = tqdm(enumerate(zip(data_loader_train,cycle(data_loader_val))), total=len(data_loader_train),desc=f'Train Epoch{epoch+1}/{epochs}: lr {optimizer.param_groups[0]["lr"]}' )
        for idx, ((source_eeg, source_bio, labels), (target_eeg, target_bio, target_labels)) in train_bar:
            #load the samples into device
            source_eeg = source_eeg.to(device)
            source_bio = source_bio.to(device)
            labels = labels.to(device)

            target_eeg = target_eeg.to(device)
            target_bio = target_bio.to(device)
            target_labels = target_labels.to(device)

            optimizer.zero_grad()

            labels = torch.argmax(labels, dim=1)
            target_labels = torch.argmax(target_labels, dim=1)
            
            (source_pred, target_pred, pred_domain,
            s_eeg_extract, s_bio_extract, t_eeg_extract, t_bio_extract,
            norm_s_eeg_extract, norm_s_bio_extract, norm_t_eeg_extract, norm_t_bio_extract
            ) = model(source_eeg, source_bio, target_eeg, target_bio)

            class_loss = class_criterion(source_pred, labels) 

            s_corr_loss = corr_criterion(s_eeg_extract, s_bio_extract)
            t_corr_loss = corr_criterion(t_eeg_extract, t_bio_extract)
            corr_loss = s_corr_loss + t_corr_loss

            finegrained_eeg_loss = finegrained_criterion(norm_s_eeg_extract, norm_t_eeg_extract)
            finegrained_bio_loss = finegrained_criterion(norm_s_bio_extract, norm_t_bio_extract)
            finegrained_loss = finegrained_eeg_loss + finegrained_bio_loss 

            source_sample_num = len(source_eeg)
            target_sample_num = len(target_eeg)
            domain_labels = torch.cat([torch.zeros(source_sample_num, 1), torch.ones(target_sample_num, 1)], dim=0).to(device)
            coarsegrained_loss = coarsegrained_criterion(pred_domain, domain_labels)

            align_loss = (finegrained_loss + coarsegrained_loss)/2

            ce_loss =  ce_criterion(target_pred)
            total_loss = class_loss + lambda_c * corr_loss + align_loss + lambda_v * ce_loss
            
            metric.update(torch.argmax(source_pred, dim=1), labels, class_criterion(source_pred, labels).item())
            train_bar.set_postfix_str(f'Classification Loss:{class_loss.item():.2f}')
            
            total_loss.backward()

            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        print('\033[32m train state:'+metric.value())
        metric_value = evaluate(model, data_loader_val, device, metrics, class_criterion)
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer,epoch, metric=m, state='best')
        '''value_now = metric_value[metric_choose]
        if value_now <= value_previous:
            count += 1
            value_previous = value_now
        else:
            count = 0 
            value_previous = value_now
        if count > 10:
            break'''
        
    model.load_state_dict(torch.load(f'{output_dir}/checkpoint-best{metric_choose}')['model'])
    if test_sub_label is not None:
        metric_value = sub_evaluate(model, data_loader_test, test_sub_label_loader, device, metrics, class_criterion, loss_func, loss_param)
    else:
        metric_value = evaluate(model, data_loader_test, device, metrics, class_criterion, loss_func, loss_param)
    for m in metrics:
        print(f'best_val_{m}: {best_metric[m]:.2f}')
        print(f'best_test_{m}: {metric_value[m]:.2f}')

    return metric_value


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, loss_func=None, loss_param=None):
    model.eval()
    metric = Metric(metrics)
    eval_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Evaluating:')

    for idx, (eeg_features, bio_features, labels) in eval_bar:
        eeg_features = eeg_features.to(device)
        bio_features = bio_features.to(device)
        labels = labels.to(device)
        labels = torch.argmax(labels, dim=1)

        (source_pred, target_pred, pred_domain,
            s_eeg_extract, s_bio_extract, t_eeg_extract, t_bio_extract,
            norm_s_eeg_extract, norm_s_bio_extract, norm_t_eeg_extract, norm_t_bio_extract
            ) = model(eeg_features, bio_features, eeg_features, bio_features)
        loss = criterion(source_pred, labels) 
        metric.update(torch.argmax(source_pred, dim=1), labels, loss.item())

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

        (source_pred, target_pred, pred_domain,
            s_eeg_extract, s_bio_extract, t_eeg_extract, t_bio_extract,
            norm_s_eeg_extract, norm_s_bio_extract, norm_t_eeg_extract, norm_t_bio_extract
            ) = model(eeg_features, bio_features, eeg_features, bio_features)

        loss = criterion(source_pred, labels) + (0 if loss_func is None else loss_func(loss_param))
        metric.update(torch.argmax(source_pred, dim =1),labels, sub_label, loss.item())
    
    print('\033[34m eval state:' + metric.value())
    return metric.values
