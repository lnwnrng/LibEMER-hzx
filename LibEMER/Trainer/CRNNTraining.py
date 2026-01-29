from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from utils.metric import Metric, SubMetric
from utils.store import save_state

def train(model, dataset_pretrain, dataset_train, dataset_val, dataset_test, device, output_dir="result/", metrics=None, metric_choose=None, optimizer=None, scheduler=None, batch_size=16, epochs=40, criterion=None,loss_func=None, loss_param= None,test_sub_label = None):
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
    test_sub_label_loader = DataLoader(test_sub_label, sampler=sampler_test, batch_size=batch_size, num_workers=4)

    model = model.to(device)

    best_metric = {s: 0.0 for s in metrics}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        metric = Metric(metrics)
        
        # 训练进度条
        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=
                         f"Train Epoch {epoch + 1}/{epochs}: lr:{optimizer.param_groups[0]['lr']}")
        
        for idx, (eeg_data, peri_data, targets) in train_bar:
            eeg_data = eeg_data.to(device)
            peri_data = peri_data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            outputs = model(eeg_data, peri_data)
            loss = criterion(outputs, targets)
            metric.update(torch.argmax(outputs, dim=1), targets, loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")
            
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        print(f"\033[32m Train state: {metric.value()}")
        
        metric_value = evaluate(model, data_loader_val, device, metrics, criterion)
        
        # 保存最佳模型
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)
    
    # 加载最佳模型
    best_checkpoint = Path(output_dir) / f'checkpoint-best{metric_choose}'
    if best_checkpoint.exists():
        checkpoint = torch.load(str(best_checkpoint))
        model.load_state_dict(checkpoint['model'])

    if test_sub_label is not None:
        metric_value = sub_evaluate(model, data_loader_test, test_sub_label_loader, device, metrics, criterion, loss_func, loss_param)
    else:
        metric_value = evaluate(model, data_loader_test, device, metrics, criterion, loss_func, loss_param)
    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")
    
    return metric_value

@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion):
    model.eval()
    
    metric = Metric(metrics)

    for idx, (eeg_data, peri_data, targets) in tqdm(enumerate(data_loader), total=len(data_loader),
                                                   desc=f"Evaluating:"):  
        eeg_data = eeg_data.to(device)
        peri_data = peri_data.to(device)
        targets = targets.to(device)  

        outputs = model(eeg_data, peri_data)
        loss = criterion(outputs, targets)
        
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())
    
    print(f"\033[34m Eval state: {metric.value()}")
    
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

        prediction = model(eeg_features, bio_features)
        loss = criterion(prediction, labels) + (0 if loss_func is None else loss_func(prediction, labels))

        metric.update(torch.argmax(prediction, dim =1), labels, sub_label, loss.item())
    
    print('\033[34m eval state:' + metric.value())
    return metric.values
