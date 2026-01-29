from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args

from data_utils.load_data import get_data
from data_utils.preprocess import preprocess, label_process, multimodal_preprocess, normalize
from data_utils.split import get_split_index, index_to_data_multimodal, merge_to_part_multimodal

from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.BimodalLSTMTraining import train

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

#seed
#python BimodalLSTM_train.py -model BimodalLSTM -seed 3407 -use_multimodal -dataset seed_de_lds -dataset_path SEED -sample_length 6 -stride 1 -bio_length 6 -bio_stride 1  -onehot -sessions 1 2 3 -split_type front_back -front 9 -lr 1e-4 -batch_size 32 -epochs 100
#84.21 / 13.37 0.7771/0.1338 2和4

#deap valence
#python DCCA_official_train.py -model DCCA_AM -use_multimodal -dataset deap -dataset_path data_preprocessed_python -bounds 5 5 -label_used valence -onehot -feature_type de_lds -time_window 2 -sample_length 1 -stride 1 -bio_length 256 -bio_stride 256 -split_type kfold -fold_num 10 -lr 1e-4 -batch_size 30 -epochs 80


#seed dependent
#CUDA_VISIBLE_DEVICES=1 nohup python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds -sample_length 6 -bio_length 6 -stride 1 -stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/BimodalLSTM/seed_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.6699/0.3198
#ALLRound Mean and Std of macro-f1 : 0.6183/0.3350

#seed indep
#CUDA_VISIBLE_DEVICES=3 nohup python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds -sessions 1  -sample_length 6 -bio_length 6 -stride 1 -stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/BimodalLSTM/seed_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.4348/0.1558
#ALLRound Mean and Std of macro-f1 : 0.3462/0.1828

#seedv depend
#CUDA_VISIBLE_DEVICES=2 nohup python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -sample_length 6 -bio_length 6 -stride 1 -stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/BimodalLSTM/seedv_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.1612/0.2531
#ALLRound Mean and Std of macro-f1 : 0.0968/0.1608

#seedv indep
#CUDA_VISIBLE_DEVICES=2 nohup python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -sessions 1 -sample_length 6 -bio_length 6 -stride 1 -stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/BimodalLSTM/seedv_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.3845/0.0833
#ALLRound Mean and Std of macro-f1 : 0.3181/0.0653

#deapv depend
#CUDA_VISIBLE_DEVICES=3 nohup python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -sample_length 5  -bio_length 640 -stride 1 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/BimodalLSTM/deapv_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5367/0.1558
#ALLRound Mean and Std of macro-f1 : 0.4707/0.1609

#deapv indep
#CUDA_VISIBLE_DEVICES=1 nohup python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -sample_length 5  -bio_length 640 -stride 1 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/BimodalLSTM/deapv_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5356/0.0751
#ALLRound Mean and Std of macro-f1 :0.3654/0.0405

#deapa depend
#CUDA_VISIBLE_DEVICES=3 nohup python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -sample_length 5  -bio_length 640 -stride 1 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/BimodalLSTM/deapa_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5919/0.1892
#ALLRound Mean and Std of macro-f1 : 0.4885/0.1967

#deapa indep
#CUDA_VISIBLE_DEVICES=2 nohup python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -sample_length 5  -bio_length 640 -stride 1 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/BimodalLSTM/deapa_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5222/0.0874
#ALLRound Mean and Std of macro-f1 : 0.4261/0.0748

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
                
                train_eeg, val_eeg, test_eeg = normalize(train_eeg, val_eeg, test_eeg, dim="sample", method="z-score")
                train_bio, val_bio, test_bio = normalize(train_bio, val_bio, test_bio, dim="sample", method="z-score")
                
                eeg_input_dim = eeg_channels * eeg_feature_dim
                bio_input_dim = bio_channels * bio_feature_dim
                eeg_hidden_size = 64
                bio_hidden_size = 32 #32
                num_layers = 2
                eeg_dropout_rate = 0.6 #0.7
                bio_dropout_rate = 0.6 #0.7               

                model = Model['BimodalLSTM'](eeg_input_dim, bio_input_dim, eeg_hidden_size,bio_hidden_size,num_layers,eeg_dropout_rate,bio_dropout_rate, num_classes)
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 1e-3)#1e-2
                criterion = nn.MultiMarginLoss(p=1, margin = 1)
                #criterion = nn.CrossEntropyLoss()

                output_dir = make_output_dir(args, 'BimodalLSTM')
                round_metric = train(model = model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    optimizer=optimizer, criterion=criterion, output_dir=output_dir, 
                                    metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs, 
                                    loss_func=None, loss_param= None, test_sub_label=test_sub_label) 
                
                
                best_metrics.append(round_metric)
                if setting.experiment_mode =='sub_dependent':
                    subjects_metrics[rridx].append(round_metric)
                    
        if setting.experiment_mode == "sub_dependent":
            sub_result_log(args, subjects_metrics)
        else:
            result_log(args, best_metrics) 
                
    elif setting.dataset.startswith('deap'):
        eeg_data, bio_data, label,eeg_channels, bio_channels, eeg_feature_dim, bio_feature_dim, num_classes = get_data(setting)
        sample_rate = 128

        #对时域信息求最大值，最小值，均值，方差，标准差，平方和
        new_bio_data = []
        for idx, session in enumerate(bio_data):
            new_session = []
            for ridx, subject in enumerate(session):
                new_subject = []
                for trial in subject:
                    trial = trial.reshape(trial.shape[0],bio_channels, -1, sample_rate)
                    trial = trial.transpose(0,2,1,3)#(sample_num,sequence_length, channels, sample_rate)

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

                train_eeg, val_eeg, test_eeg = normalize(train_eeg, val_eeg, test_eeg, dim="sample", method="z-score")
                train_bio, val_bio, test_bio = normalize(train_bio, val_bio, test_bio, dim="sample", method="z-score")

                eeg_input_dim = eeg_channels * eeg_feature_dim
                bio_input_dim = bio_channels * 6
                eeg_hidden_size = 64
                bio_hidden_size = 32 #32
                num_layers = 2
                eeg_dropout_rate = 0.6 #0.7
                bio_dropout_rate = 0.6 #0.7 

                model = Model['BimodalLSTM'](eeg_input_dim, bio_input_dim, eeg_hidden_size,bio_hidden_size,num_layers,eeg_dropout_rate,bio_dropout_rate, num_classes)
            
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 1e-3)#1e-2
                criterion = nn.MultiMarginLoss(p=1, margin = 1)
                #criterion = nn.CrossEntropyLoss()

                output_dir = make_output_dir(args, 'BimodalLSTM')
                round_metric = train(model = model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    optimizer=optimizer, criterion=criterion, output_dir=output_dir, 
                                    metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs, 
                                    loss_func=None, loss_param= None,test_sub_label=test_sub_label) 
                
                
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
