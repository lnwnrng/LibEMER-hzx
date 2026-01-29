from config.setting import set_setting_by_args, preset_setting

from data_utils.load_data import get_data
from data_utils.preprocess import preprocess, label_process, multimodal_preprocess
from data_utils.split import get_split_index, index_to_data_multimodal, merge_to_part_multimodal

from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from models.Models import Model
from models.CFDA_CSF import CORALLoss, CMDLoss, CELoss
from Trainer.CFDA_CSFTraining import train
#seedv
#CUDA_VISIBLE_DEVICES=3 nohup python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -dataset seedv_de_lds -dataset_path SEEDV -experiment_mode sub_independent -onehot -sessions 1 -sample_length 1 -stride 1 -bio_length 1 -bio_stride 1 -split_type leave_one_out -lr 5e-4 -batch_size 64 -epochs 100 >CFDA_CSF/seedv_reproduction_session1.log
#seed
#python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -dataset seed_de_lds -dataset_path SEED -experiment_mode sub_independent -sample_length 1 -stride 1 -bio_length 1 -bio_stride 1  -onehot -sessions 1 2 3 -split_type leave_one_out 
#deap de_feature
#python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -dataset deap -dataset_path data_preprocessed_python -bounds 5 5 -label_used valence -onehot -feature_type de_lds -time_window 2 -sample_length 1 -stride 1 -bio_length 256 -bio_stride 256 -split_type kfold -fold_num 10 -lr 1e-4 -batch_size 30 -epochs 80

#seed dependent
#CUDA_VISIBLE_DEVICES=3 nohup python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >CFDA_CSF/seed_dependent_train_val_test_lr1e-4.log
#0.8103/0.2024 0.7585/0.2541

#seed independent
#CUDA_VISIBLE_DEVICES=1 nohup python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds   -batch_size 64 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/CFDA_CSF/seed_independent_train_val_test_session1_lr1e-4.log
#0.3806/0.0020  0.3272/0.0480 

#seedv dependent
#CUDA_VISIBLE_DEVICES=1 nohup python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 5e-4 -onehot >CFDA_CSF/seedv_dependent_train_val_test_lr5e-4.log
#ALLRound Mean and Std of acc : 0.2285/0.2309
#ALLRound Mean and Std of macro-f1 : 0.1577/0.1580

#seedv independent
#CUDA_VISIBLE_DEVICES=0 nohup python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 5e-4 -onehot >/data1/cxx/CFDA_CSF/seedv_independent_train_val_test_session1_lr5e-4.log
#session1 0.4557/0.2364 0.4644/0.1912
#session2 0.3734 0.3028

#deapv dependent
#CUDA_VISIBLE_DEVICES=0 nohup python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 64 -epochs 100 -lr 1e-4 >CFDA_CSF/deapv_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5557/0.1479
#ALLRound Mean and Std of macro-f1 : 0.4935/0.1482

#deapv indep
#CUDA_VISIBLE_DEVICES=1 nohup python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 64 -epochs 100 -lr 1e-4 >/data1/cxx/CFDA_CSF/deapv_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5241/0.0941
#ALLRound Mean and Std of macro-f1 : 0.4584/0.1264

#deapa dependent
#CUDA_VISIBLE_DEVICES=2 nohup python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 64 -epochs 100 -lr 1e-4 >CFDA_CSF/deapa_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5513/0.1949
#ALLRound Mean and Std of macro-f1 : 0.4565/0.1788

#deapa indep
#CUDA_VISIBLE_DEVICES=1 nohup python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 64 -epochs 100 -lr 1e-4 >/data1/cxx/CFDA_CSF/deapa_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5083/ 0.1191
#ALLRound Mean and Std of macro-f1 : 0.5006/0.0496

def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setup_seed(args.seed)
    #如果要提取频域特征，需要手动设置提取的频段
    '''setting.extract_bands = [[4,8],[8,10],[8,12],[12,30],[30,45]]
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
        #print(len(eeg_data))
        best_metrics = []
        subjects_metrics = [[]for _ in range(len(eeg_data))]
        #print(len(subjects_metrics))
        
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
                print(f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')#EEG: 62 *5 PPS:1*33
                print(f'test_eeg_data shape:{test_eeg.shape}, test_bio_data shape:{test_bio.shape}, test_label shape:{test_label.shape}')

                if len(val_eeg) == 0:
                    val_eeg = test_eeg
                    val_bio = test_bio
                    val_label = test_label
                
                eeg_input_dim = eeg_channels * eeg_feature_dim
                bio_input_dim = bio_channels * bio_feature_dim
                dropout_rate=0.5
                if not setting.dataset.startswith(('seediv', 'seedv')):
                    lambda_c = 0.5
                else:
                    lambda_c = 0.1
                lambda_v = 0.1               

                model = Model['CFDA_CSF'](eeg_input_dim, bio_input_dim, dropout_rate,num_classes)
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 1e-4)#1e-2
                class_criterion = nn.CrossEntropyLoss()
                corr_criterion = CORALLoss()
                finegrained_criterion = CMDLoss(k=2)
                coarsegrained_criterion = nn.BCELoss()
                ce_criterion = CELoss(threshold=0.5)

                output_dir = make_output_dir(args, 'CFDA_CSF')
                round_metric = train(model = model,  dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    optimizer=optimizer, output_dir=output_dir, metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs, 
                                    class_criterion = class_criterion, corr_criterion=corr_criterion, finegrained_criterion=finegrained_criterion, 
                                    coarsegrained_criterion=coarsegrained_criterion, ce_criterion=ce_criterion, lambda_c=lambda_c, lambda_v=lambda_v,loss_func=None, loss_param= None,test_sub_label=test_sub_label)
                
                best_metrics.append(round_metric)
                if setting.experiment_mode =='sub_dependent':
                    subjects_metrics[rridx].append(round_metric)
                    
        if setting.experiment_mode == "sub_dependent":
            sub_result_log(args, subjects_metrics)
        else:
            result_log(args, best_metrics) 
                
    elif setting.dataset.startswith('deap'):
        eeg_data, bio_data, label,eeg_channels, bio_channels, eeg_feature_dim, bio_feature_dim, num_classes = get_data(setting)

        #对时域信息求最大值，最小值，均值，方差，标准差，平方和
        new_bio_data = []
        for idx, session in enumerate(bio_data):
            new_session = []
            for ridx, subject in enumerate(session):
                new_subject = []
                for trial in subject:
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
                print(f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')# EEG:32 *5, pps:8*6
                print(f'test_eeg_data shape:{test_eeg.shape}, test_bio_data shape:{test_bio.shape}, test_label shape:{test_label.shape}')

                if len(val_eeg) == 0:
                    val_eeg = test_eeg
                    val_bio = test_bio
                    val_label = test_label

                eeg_input_dim =  eeg_channels * eeg_feature_dim
                bio_input_dim = bio_channels *6
                dropout_rate = 0.5
                lambda_c = 0.5
                lambda_v = 0.1

                model = Model['CFDA_CSF'](eeg_input_dim, bio_input_dim, dropout_rate,num_classes)
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 1e-4)#1e-2
                class_criterion = nn.CrossEntropyLoss()
                corr_criterion = CORALLoss()
                finegrained_criterion = CMDLoss(k=2)
                coarsegrained_criterion = nn.BCELoss()
                ce_criterion = CELoss(threshold=0.5)

                output_dir = make_output_dir(args, 'CFDA_CSF')
                round_metric = train(model = model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    optimizer=optimizer, output_dir=output_dir, metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs, 
                                    class_criterion = class_criterion, corr_criterion=corr_criterion, finegrained_criterion=finegrained_criterion, 
                                    coarsegrained_criterion=coarsegrained_criterion, ce_criterion=ce_criterion, lambda_c=lambda_c, lambda_v=lambda_v,loss_func=None, loss_param= None,
                                    test_sub_label=test_sub_label)
                
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