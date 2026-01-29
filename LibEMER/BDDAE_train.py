from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args

from data_utils.load_data import get_data
from data_utils.preprocess import preprocess, label_process, multimodal_preprocess, normalize
from data_utils.split import get_split_index, index_to_data_multimodal, merge_to_part_multimodal

from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.BDDAETraining import train

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
#seed
#CUDA_VISIBLE_DEVICES=1 nohup python BDDAE_train.py -model BDDAE -use_multimodal -dataset seed_de_lds -dataset_path SEED -experiment_mode sub_dependent -sample_length 1 -stride 1 -bio_length 1 -bio_stride 1  -onehot -sessions 1 2 3 -split_type front_back -front 9 -lr 1e-3 -batch_size 32 -epochs 200 >BDDAE/SEED_reproduction.log
#0.7937/0.0892 lr1e-3 weight_decay 1e-3

#seed dependent
#CUDA_VISIBLE_DEVICES=2 nohup python BDDAE_train.py -model BDDAE -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 200 -lr 1e-3 -onehot >BDDAE/seed_dependent_train_val_test_lr1e-3.log
# ALLRound Mean and Std of acc : 0.7923/0.1946
# ALLRound Mean and Std of macro-f1 : 0.7407/0.2441

#seed indep
#CUDA_VISIBLE_DEVICES=0 nohup python BDDAE_train.py -model BDDAE -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 200 -lr 1e-3 -onehot >/data1/cxx/BDDAE/seed_independent_train_val_test_session1_lr1e-3.log
#ALLRound Mean and Std of acc : 0.5689/0.0689
#ALLRound Mean and Std of macro-f1 : 0.4892/0.0026

#seedv dependent
#CUDA_VISIBLE_DEVICES=2 nohup python BDDAE_train.py -model BDDAE -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 200 -lr 1e-3 -onehot >/data1/cxx/BDDAE/seedv_dependent_train_val_test_lr1e-3.log
#ALLRound Mean and Std of acc : 0.1694/0.2199
#ALLRound Mean and Std of macro-f1 : 0.1130/0.1479

#seedv indep
#CUDA_VISIBLE_DEVICES=2 nohup python BDDAE_train.py -model BDDAE -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 200 -lr 1e-3 -onehot >/data1/cxx/BDDAE/seedv_independent_train_val_test_session1_lr1e-3.log
#ALLRound Mean and Std of acc : 0.2511/0.0852
#ALLRound Mean and Std of macro-f1 : 0.1787/0.1193

#deapv depend
#CUDA_VISIBLE_DEVICES=0 nohup python BDDAE_train.py -model BDDAE -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 200 -lr 1e-3 >/data1/cxx/BDDAE/deapv_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5560/0.1688
#ALLRound Mean and Std of macro-f1 : 0.4684/0.1726

#deapv indep
#CUDA_VISIBLE_DEVICES=2 nohup python BDDAE_train.py -model BDDAE -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 200 -lr 1e-4 >/data1/cxx/BDDAE/deapv_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5583/0.0516 
#ALLRound Mean and Std of macro-f1 : 0.3577/0.0195

#deapa depend
#CUDA_VISIBLE_DEVICES=2 nohup python BDDAE_train.py -model BDDAE -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 200 -lr 1e-3 >/data1/cxx/BDDAE/deapa_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5890/0.1493
#ALLRound Mean and Std of macro-f1 : 0.4591/0.1286

#deapa indep
#CUDA_VISIBLE_DEVICES=2 nohup python BDDAE_train.py -model BDDAE -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 200 -lr 1e-4 >/data1/cxx/BDDAE/deapa_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc :  0.5458/0.0914
#ALLRound Mean and Std of macro-f1 :  0.3512/0.0346

def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setup_seed(args.seed)
    #如果要提取频域特征，需要手动设置提取的频段
    '''setting.extract_bands = [[4,8],[8,14],[14,31],[31,45]]
    setting.eog_bands = [[4,8],[8,14],[14,31],[31,45]]
    setting.emg_bands = [[4,8],[8,14],[14,31],[31,45]]
    setting.gsr_bands = [[0, 0.6],[0.6, 1.2],[1.2, 1.8],[1.8, 2.4]]
    setting.bvp_bands = [[0,0.1],[0.1,0.2],[0.2,0.3],[0.3,0.4]]
    setting.resp_bands = [[0,0.6],[0.6,1.2],[1.2,1.8],[1.8,2.4]]
    setting.temp_bands =[[0, 0.05],[0.05, 0.1],[0.1, 0.15],[0.15, 0.2]]'''

    device  = torch.device(args.device)
    assert setting.use_multimodal == True, 'You do not use multimodal data, please set use_multimodal to True'
    if setting.dataset.startswith('seed'): 
        all_eeg, all_bio, all_label,eeg_channels, bio_channels, eeg_feature_dim, bio_feature_dim, num_classes = get_data(setting)
        eeg_data, bio_data, label = merge_to_part_multimodal(all_eeg, all_bio, all_label,setting)
        print(len(eeg_data))
        best_metrics = []
        subjects_metrics = [[]for _ in range(len(eeg_data))]
        print(len(subjects_metrics))
        eeg_dim =  eeg_channels * eeg_feature_dim
        bio_dim = bio_channels * bio_feature_dim
        h_eeg_dim = 128
        h_bio_dim = 32
        joint_dim = 128 
        eeg_dropout_rate = 0.5 
        bio_dropout_rate = 0.5 
        for rridx, (eeg_data_i, bio_data_i, label_i) in enumerate(zip(eeg_data, bio_data, label)):
            tts = get_split_index(eeg_data_i, label_i, setting)
            
            for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(\
                zip(tts['train'], tts['test'], tts['val'])):
                setup_seed(args.seed)
                if val_indexes[0] == -1:
                    print(f'train indexes:{train_indexes}, test indexes:{test_indexes}')
                else:
                    print(f'train indexes:{train_indexes}, test indexes:{test_indexes}, val indexes:{val_indexes}')

                test_sub_label = None
                if setting.experiment_mode == 'sub_independent':
                    train_eeg, train_bio,train_label, val_eeg,val_bio,val_label, test_eeg,test_bio, test_label =\
                    index_to_data_multimodal(eeg_data_i, bio_data_i, label_i,train_indexes,test_indexes,val_indexes,True)
                    test_sub_num = len(test_eeg)
                    test_sub_label = []
                    for i in range(test_sub_num):
                        test_sub_count = len(test_eeg[i])
                        test_sub_label.extend([i+1 for j in range(test_sub_count)])
                    test_sub_label = np.array(test_sub_label)
                print(test_sub_label)

                train_eeg, train_bio,train_label, val_eeg,val_bio,val_label, test_eeg,test_bio, test_label =\
                    index_to_data_multimodal(eeg_data_i, bio_data_i, label_i,train_indexes,test_indexes,val_indexes,keep_dim=args.keep_dim)
                print(f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')
                print(f'test_eeg_data shape:{test_eeg.shape}, test_bio_data shape:{test_bio.shape}, test_label shape:{test_label.shape}')

                if len(val_eeg) == 0:
                    val_eeg = test_eeg
                    val_bio = test_bio
                    val_label = test_label

                train_eeg, val_eeg, test_eeg = normalize(train_eeg, val_eeg, test_eeg, dim="sample", method="minmax")
                train_bio, val_bio, test_bio = normalize(train_bio, val_bio, test_bio, dim="sample", method="minmax")
                   
                model = Model['BDDAE'](eeg_dim, bio_dim, h_eeg_dim,h_bio_dim,joint_dim,eeg_dropout_rate,bio_dropout_rate, num_classes)
                pretrain_dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio))
                pretrain_dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio))
                dataset_pretrain = torch.utils.data.ConcatDataset([pretrain_dataset_train, pretrain_dataset_val])
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))
                
                pretrain_optimizer = optim.AdamW(model.encoder_parameters(), lr=args.lr)
                finetune_optimizer_param_groups = [
                    {'params': model.encoder_parameters(), 'lr': args.lr / 100}, # 编码器使用1%的学习率
                    {'params': model.classifier_parameters(), 'lr': args.lr}      # 分类器使用100%的学习率 
                ]
                finetune_optimizer = optim.AdamW(finetune_optimizer_param_groups, weight_decay = 1e-3)
            
                recon_criterion = nn.MSELoss()
                class_criterion = nn.MultiMarginLoss(p=1, margin = 1)

                output_dir = make_output_dir(args, 'BDDAE')
                round_metric = train(model = model, dataset_pretrain=dataset_pretrain,dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    pretrain_optimizer=pretrain_optimizer, finetune_optimizer=finetune_optimizer,recon_criterion=recon_criterion, class_criterion=class_criterion, output_dir=output_dir, 
                                    metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs, 
                                    loss_func=None, loss_param= None,test_sub_label=test_sub_label) 
                
                best_metrics.append(round_metric)
                if setting.experiment_mode =='sub_dependent':
                    subjects_metrics[rridx].append(round_metric)
                    

        if setting.experiment_mode == "sub_dependent":
            sub_result_log(args, subjects_metrics)
        else:
            result_log(args, best_metrics) 



    else:  
        eeg_data, bio_data, label,eeg_channels, bio_channels, eeg_feature_dim, bio_feature_dim, num_classes = get_data(setting)

        #对时域信息求最大值，最小值，均值，方差，标准差，平方和
        new_bio_data = []
        for idx, session in enumerate(bio_data):
            new_session = []
            for ridx, subject in enumerate(session):
                new_subject = []
                for trial in subject:
                    trial = trial.reshape(trial.shape[0],trial.shape[1], -1,128)
                    trial = np.transpose(trial, (0,2,1,3))
                    mean = trial.mean(axis=-1)
                    std = trial.std(axis=-1)
                    max = trial.max(axis=-1)
                    min = trial.min(axis=-1)
                    sq_sum = (trial**2).sum(axis=-1)
                    var = np.var(trial, axis=-1)
                    std = np.sqrt(var)
                    new_trial = np.stack([min, max, mean, std, var, sq_sum], axis=-1)
                    new_subject.append(new_trial)
                new_session.append(new_subject)
            new_bio_data.append(new_session)


        eeg_data, bio_data, label = merge_to_part_multimodal(eeg_data, new_bio_data, label,setting)
        
        best_metrics =[]
        subjects_metrics = [[]for _ in range(len(eeg_data))]
        eeg_dim =  eeg_channels * eeg_feature_dim
        bio_dim = bio_channels * 6
        h_eeg_dim = 128
        h_bio_dim = 32
        joint_dim = 128
        eeg_dropout_rate = 0.5
        bio_dropout_rate = 0.5 
        for rridx, (eeg_data_i, bio_data_i, label_i) in enumerate(zip(eeg_data, bio_data, label)):
            tts = get_split_index(eeg_data_i, label_i, setting)
            for ridx,(train_indexes, test_indexes, val_indexes)  in enumerate(zip(tts['train'], tts['test'], tts['val'])):
                setup_seed(args.seed)
                if val_indexes[0] == -1:
                    print(f'train indexes:{train_indexes}, test indexes:{test_indexes}')
                else:
                    print(f'train indexes:{train_indexes}, test indexes:{test_indexes}, val indexes:{val_indexes}')

                test_sub_label = None
                if setting.experiment_mode == 'sub_independent':
                    train_eeg, train_bio,train_label, val_eeg,val_bio,val_label, test_eeg,test_bio, test_label =\
                    index_to_data_multimodal(eeg_data_i, bio_data_i, label_i,train_indexes,test_indexes,val_indexes,True)
                    test_sub_num = len(test_eeg)
                    test_sub_label = []
                    for i in range(test_sub_num):
                        test_sub_count = len(test_eeg[i])
                        test_sub_label.extend([i+1 for j in range(test_sub_count)])
                    test_sub_label = np.array(test_sub_label)
                print(test_sub_label)
                
                train_eeg, train_bio,train_label, val_eeg,val_bio,val_label, test_eeg,test_bio, test_label =\
                    index_to_data_multimodal(eeg_data_i, bio_data_i, label_i,train_indexes,test_indexes,val_indexes,keep_dim=args.keep_dim)
                print(f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')
                print(f'test_eeg_data shape:{test_eeg.shape}, test_bio_data shape:{test_bio.shape}, test_label shape:{test_label.shape}')

                if len(val_eeg) == 0:
                    val_eeg = test_eeg
                    val_bio = test_bio
                    val_label = test_label
                
                train_eeg, val_eeg, test_eeg = normalize(train_eeg, val_eeg, test_eeg, dim="sample", method="minmax")
                train_bio, val_bio, test_bio = normalize(train_bio, val_bio, test_bio, dim="sample", method="minmax")

                model = Model['BDDAE'](eeg_dim, bio_dim, h_eeg_dim,h_bio_dim,joint_dim,eeg_dropout_rate,bio_dropout_rate, num_classes)
                pretrain_dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio))
                pretrain_dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio))
                dataset_pretrain = torch.utils.data.ConcatDataset([pretrain_dataset_train, pretrain_dataset_val])
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))
                
                pretrain_optimizer = optim.AdamW(model.encoder_parameters(), lr=args.lr)
                finetune_optimizer_param_groups = [
                    {'params': model.encoder_parameters(), 'lr': args.lr / 100.0}, # 编码器使用1%的学习率
                    {'params': model.classifier_parameters(), 'lr': args.lr}      # 分类器使用100%的学习率
                ]
                finetune_optimizer = optim.AdamW(finetune_optimizer_param_groups, weight_decay = 1e-3)
            
                recon_criterion = nn.MSELoss()
                class_criterion = nn.MultiMarginLoss(p=1, margin = 1)

                output_dir = make_output_dir(args, 'BDDAE')
                round_metric = train(model = model, dataset_pretrain=dataset_pretrain,dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    pretrain_optimizer=pretrain_optimizer, finetune_optimizer=finetune_optimizer,recon_criterion=recon_criterion, class_criterion=class_criterion, output_dir=output_dir, 
                                    metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs, 
                                    loss_func=None, loss_param= None, test_sub_label = test_sub_label)
                
                best_metrics.append(round_metric)
                if setting.experiment_mode =='sub_dependent':
                    subjects_metrics[rridx].append(round_metric)
            

        if setting.experiment_mode == "sub_dependent":
            sub_result_log(args, subjects_metrics)
        else:
            result_log(args, best_metrics)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    state_log(args)
    main(args)