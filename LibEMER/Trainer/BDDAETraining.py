import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from tqdm import tqdm
from itertools import cycle


from utils.metric import Metric, SubMetric
from utils.store import save_state

class LossMetric:
    def __init__(self):
        
        self.avgloss = 0
        self.losses =[]

    def update(self, loss):
        self.losses.append(loss)

    def loss(self):
        self.avgloss = sum(self.losses)/len(self.losses)
        return self.avgloss
    def value(self):
        out = ''
        func = {
            'loss': self.loss
        }
        
        out = f"loss:{func['loss']():.3f}"
        return out
def train(model, dataset_pretrain,dataset_train, dataset_val, dataset_test, device, output_dir='result/', metrics= None, metric_choose = None, pretrain_optimizer = None, finetune_optimizer= None, scheduler=None,batch_size = 16, epochs=40, 
          recon_criterion = None, class_criterion = None, loss_func = None, loss_param= None, test_sub_label = None ):
    if metrics is None:
        metrics=['acc']
    if metric_choose is None:
        metric_choose = metrics[0]

    loss_metics = ['loss']
    
    #data sampler for train, val, test
    sampler_pretrain = RandomSampler(dataset_pretrain)
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)
    
    data_loader_pretrain = DataLoader(dataset_pretrain, sampler= sampler_pretrain, batch_size = batch_size, num_workers =4)
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4)
    data_loader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4)
    test_sub_label_loader = DataLoader(
        test_sub_label, sampler=sampler_test, batch_size=batch_size, num_workers=4
    ) if test_sub_label is not None else None
    model = model.to(device)

    best_metric = {s:0. for s in metrics}
    best_loss = 0
    value_previous = 0
    count = 0
    pretrain_epochs = epochs // 2
    finetune_epochs = epochs - pretrain_epochs
    print('-----Start Pretraining-----')
    for epoch in range(pretrain_epochs):
        model.train()
        lossmetric = LossMetric()
        
        pretrain_optimizer.zero_grad()
        train_bar = tqdm(enumerate(data_loader_pretrain), total=len(data_loader_pretrain),desc=f'PreTrain Epoch{epoch+1}/{pretrain_epochs}: lr {pretrain_optimizer.param_groups[0]["lr"]}' )
        for idx, (eeg_features, bio_features) in train_bar:
            #load the samples into device
            eeg_features = eeg_features.to(device)
            bio_features = bio_features.to(device)
            
            eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
            bio_features = bio_features.reshape(bio_features.shape[0], -1)

            pretrain_optimizer.zero_grad()

            recon_eeg, recon_bio, prediction = model(eeg_features, bio_features)
            loss_eeg = recon_criterion(recon_eeg, eeg_features)
            loss_bio = recon_criterion(recon_bio, bio_features)
            recon_loss = loss_eeg + loss_bio
            
            lossmetric.update(recon_loss.item())
            train_bar.set_postfix_str(f'Reconstruction Loss:{recon_loss.item():.2f}')
            
            recon_loss.backward()

            pretrain_optimizer.step()

        if scheduler is not None:
            scheduler.step()
        print('\033[32m train state:'+lossmetric.value())
        loss_value = pretrain_evaluate(model, data_loader_val, device, recon_criterion, loss_func, loss_param)
        if epoch == 0 or loss_value < best_loss:
            best_loss = loss_value
            save_state(output_dir, model, pretrain_optimizer, epoch, metric='pretrainloss', state='best')
   
    model.load_state_dict(torch.load(f'{output_dir}/checkpoint-bestpretrainloss')['model'])
    print('-----Finetuning-----')
    for epoch in range(finetune_epochs):
        model.train()
        finetune_optimizer.zero_grad()
        metric = Metric(metrics)
        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train),desc=f'Finetune Epoch{epoch+1}/{finetune_epochs}: lr {finetune_optimizer.param_groups[0]["lr"]}' )
        for idx, (eeg_features, bio_features, labels) in train_bar:
            #load the samples into device
            eeg_features = eeg_features.to(device)
            bio_features = bio_features.to(device)
            eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
            bio_features = bio_features.reshape(bio_features.shape[0], -1)
            labels = labels.to(device)

            finetune_optimizer.zero_grad()
            labels = torch.argmax(labels, dim=1)
            recon_eeg, recon_bio, prediction= model(eeg_features, bio_features)
            predict_loss = class_criterion(prediction, labels)

            metric.update(torch.argmax(prediction, dim=1), labels, predict_loss.item())
            train_bar.set_postfix_str(f'Classification Loss:{predict_loss.item():.2f}')
            
            predict_loss.backward()
            finetune_optimizer.step()
        if scheduler is not None:
            scheduler.step()
        print('\033[32m train state:'+metric.value())
        metric_value= evaluate(model, data_loader_val, device, metrics, class_criterion, loss_func, loss_param)
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, finetune_optimizer,epoch, metric=m, state='best')

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
def pretrain_evaluate(model, data_loader, device,  criterion, loss_func, loss_param):
    model.eval()
    lossmetric = LossMetric()
    eval_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Evaluating:')

    for idx, (eeg_features, bio_features, labels) in eval_bar:
        eeg_features = eeg_features.to(device)
        bio_features = bio_features.to(device)
        
        labels = labels.to(device)
        labels = torch.argmax(labels, dim=1)
        eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
        bio_features = bio_features.reshape(bio_features.shape[0], -1)
        recon_eeg, recon_bio, prediction= model(eeg_features, bio_features)
        loss_eeg = criterion(recon_eeg, eeg_features)
        loss_bio = criterion(recon_bio, bio_features)
        total_loss = loss_eeg + loss_bio        
        lossmetric.update(total_loss.item())

    print('\033[34m eval state:' + lossmetric.value())
    return lossmetric.avgloss
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
        eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
        bio_features = bio_features.reshape(bio_features.shape[0], -1)

        recon_eeg, recon_bio, prediction = model(eeg_features, bio_features)
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
        eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
        bio_features = bio_features.reshape(bio_features.shape[0], -1)

        recon_eeg, recon_bio, prediction = model(eeg_features, bio_features)
        loss = criterion(prediction, labels) + (0 if loss_func is None else loss_func(loss_param))

        metric.update(torch.argmax(prediction, dim =1),labels, sub_label, loss.item())
    
    print('\033[34m eval state:' + metric.value())
    return metric.values
