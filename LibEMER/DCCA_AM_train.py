from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args

from data_utils.load_data import get_data
from data_utils.preprocess import preprocess, label_process, multimodal_preprocess, normalize
from data_utils.split import get_split_index, index_to_data_multimodal, merge_to_part_multimodal

from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.DCCA_AMTraining import train

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

#seed
#python DCCA_AM_train.py -model DCCA_AM -use_multimodal -dataset_path SEED -lr 1e-4 -batch_size 100 -epochs 80
#88 / 12
#python DCCA_AM_train.py -model DCCA_AM -use_multimodal -dataset_path SEED -lr 0.001 -batch_size 100 -epochs 80
#python DCCA_AM_train.py -model DCCA_AM -use_multimodal -dataset_path SEED -lr 1e-4 -batch_size 30 -epochs 100
#0.9097/0.1100

#deap valence
#python DCCA_AM_train.py -model DCCA_AM -use_multimodal -dataset deap -dataset_path data_preprocessed_python -bounds 5 5 -label_used valence -onehot -time_window 2 -sample_length 1 -stride 1 -bio_length 256 -bio_stride 256 -split_type kfold -fold_num 10 -lr 1e-4 -batch_size 30 -epochs 80
#0.8732/0.0484
#deap arousal
 
#seedv
#python DCCA_AM_train.py -model DCCA_AM -use_multimodal -dataset seedv_de_lds -dataset_path SEEDV -sessions 1 2 3  -onehot  -sample_length 1 -stride 1 -bio_length 1 -bio_stride 1 -split_type kfold -fold_num 3 -fold_shuffle false -lr 1e-3 -batch_size 30 -epochs 100 
# 0.7384/0.0777

#seed depend
#CUDA_VISIBLE_DEVICES=1 nohup python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/DCCA_AM/seed_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.8201/0.1972
#ALLRound Mean and Std of macro-f1 : 0.7764/0.2387

#seed indep
#CUDA_VISIBLE_DEVICES=0 nohup python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/DCCA_AM/seed_independent_train_val_test_session1_lr1e-4.log
#ALLRound Mean and Std of acc : 0.4400/0.2259
#ALLRound Mean and Std of macro-f1 : 0.3717/0.2088

#seedv depend
#CUDA_VISIBLE_DEVICES=1 nohup python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/DCCA_AM/seedv_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.3538/0.2730
#ALLRound Mean and Std of macro-f1 : 0.2384/0.2110

#seedv indep
#CUDA_VISIBLE_DEVICES=1 nohup python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/DCCA_AM/seedv_independent_train_val_test_session1_lr1e-4.log
#ALLRound Mean and Std of acc : 0.3265/0.1785
#ALLRound Mean and Std of macro-f1 : 0.2150/0.1892

#deapv depend
#CUDA_VISIBLE_DEVICES=2 nohup python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/DCCA_AM/deapv_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5241/0.1571
#ALLRound Mean and Std of macro-f1 : 0.4870/0.1674

#deapv indep
#CUDA_VISIBLE_DEVICES=0 nohup python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/DCCA_AM/deapv_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5524/0.0867
#ALLRound Mean and Std of macro-f1 : 0.4578/0.1102

#deapa depend
#CUDA_VISIBLE_DEVICES=1 nohup python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/DCCA_AM/deapa_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5818/0.1470
#ALLRound Mean and Std of macro-f1 : 0.5104/0.1396

#deapa indep
#CUDA_VISIBLE_DEVICES=3 nohup python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/DCCA_AM/deapa_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.4699/0.0767
#ALLRound Mean and Std of macro-f1 : 0.4177/0.0673

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
    all_layer_sizes= {
        'seed': [389,218,118,20],
        'deap': [1500, 750, 500, 375, 130, 65, 30]
    }
    if setting.dataset.startswith('seed'):
        all_eeg, all_bio, all_label,eeg_channels, bio_channels, eeg_feature_dim, bio_feature_dim, num_classes = get_data(setting)
        eeg_data, bio_data, label = merge_to_part_multimodal(all_eeg, all_bio, all_label,setting)
        best_metrics = []
        subjects_metrics = [[]for _ in range(len(eeg_data))]

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
                
                input_size1 =  train_eeg.shape[1]*train_eeg.shape[2]
                input_size2 = train_bio.shape[1]*train_bio.shape[2]
                num_classes = train_label.shape[1]
                '''if not setting.dataset.startswith('seedv'):
                    layer_sizes = all_layer_sizes['seed']
                else:
                    layer_sizes = [np.random.randint(100,200), np.random.randint(20,50), 12]'''
                layer_sizes = [np.random.randint(100,200), np.random.randint(20,50), 12]
                outdim_size = layer_sizes[-1]
                r1 = 1e-6#1e-3
                r2 = 1e-6#1e-3

                model = Model['DCCA_AM'](input_size1, input_size2,layer_sizes, layer_sizes,outdim_size, r1, r2, num_classes, device)
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))

                optimizer1 = optim.RMSprop(model.model1_parameters, lr=args.lr/2)
                optimizer2 = optim.RMSprop(model.model2_parameters, lr=args.lr/2 )
                optimizer3 = optim.RMSprop(model.parameters(), lr = args.lr, weight_decay=1e-2)
                #criterion = nn.MultiMarginLoss(p=1, margin = 10)
                criterion = nn.CrossEntropyLoss()

                output_dir = make_output_dir(args, 'DCCA_AM')
                round_metric = train(model = model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    optimizer1=optimizer1, optimizer2 = optimizer2, optimizer3=optimizer3, criterion=criterion, output_dir=output_dir, 
                                    metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs, 
                                    loss_func=None, loss_param= None,test_sub_label=test_sub_label) 
                
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
                print(f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')

                if len(val_eeg) == 0:
                    val_eeg = test_eeg
                    val_bio = test_bio
                    val_label = test_label

                train_eeg, val_eeg, test_eeg = normalize(train_eeg, val_eeg, test_eeg, dim="sample", method="minmax")
                train_bio, val_bio, test_bio = normalize(train_bio, val_bio, test_bio, dim="sample", method="minmax")

                input_size1 =  train_eeg.shape[1] * train_eeg.shape[2]
                input_size2 = train_bio.shape[1]*train_bio.shape[2]
                num_classes = train_label.shape[1]
                layer_sizes = [np.random.randint(100,200), np.random.randint(20,50), 12]
                outdim_size = layer_sizes[-1]
                r1 = 1e-6
                r2 = 1e-6

                model = Model['DCCA_AM'](input_size1, input_size2,layer_sizes, layer_sizes,outdim_size, r1, r2, num_classes, device)
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))

                optimizer1 = optim.RMSprop(model.model1_parameters, lr=args.lr/2)
                optimizer2 = optim.RMSprop(model.model2_parameters, lr=args.lr/2)
                optimizer3 = optim.RMSprop(model.parameters(), lr = args.lr, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()

                output_dir = make_output_dir(args, 'DCCA_AM')
                round_metric = train(model = model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    optimizer1=optimizer1, optimizer2 = optimizer2, optimizer3=optimizer3, criterion=criterion, output_dir=output_dir, 
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
