from config.setting import set_setting_by_args, preset_setting

from data_utils.load_data import get_data
from data_utils.preprocess import preprocess, label_process, multimodal_preprocess, map_channels_to_grid
from data_utils.split import get_split_index, index_to_data_multimodal, merge_to_part_multimodal
from data_utils.constants.seed import SEED_CHANNEL_NAME, SEED_2D_GRID_LOC
from data_utils.constants.deap import DEAP_CHANNEL_NAME, DEAP_2D_GRID_LOC

from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log

from models.CRNN import CRNN
from Trainer.CRNNTraining import train

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

# python CRNN_train.py -model CRNN -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -use_multimodal -only_seg -sample_length 128 -stride 128 -bio_length 128 -bio_stride 128 -label_used valence -onehot -bounds 5 5 -split_type leave_one_out -batch_size 32 -epochs 40 -lr 0.1
# arousal - ALLRound Mean and Std of acc : 0.8835/0.0426
# valence - ALLRound Mean and Std of acc : 0.8705/0.0466

#seed dependent
#CUDA_VISIBLE_DEVICES=3 nohup python CRNN_train.py -model CRNN -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path SEED -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 0.0001 -onehot >/data1/cxx/CRNN/seed_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5956/0.1877
#ALLRound Mean and Std of macro-f1 : 0.4905/0.2234

#seed indep
#CUDA_VISIBLE_DEVICES=0 nohup python CRNN_train.py -model CRNN -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/CRNN/seed_independent_train_val_test_session1_lr1e-4.log
#ALLRound Mean and Std of acc : 0.3884/0.0890
#ALLRound Mean and Std of macro-f1 : 0.2769/0.1132

#seedv depend
#CUDA_VISIBLE_DEVICES=3 nohup python CRNN_train.py -model CRNN -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/CRNN/seedv_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.1400/0.1957
#ALLRound Mean and Std of macro-f1 : 0.1002/0.1413

#seedv indep
#CUDA_VISIBLE_DEVICES=0 nohup python CRNN_train.py -model CRNN -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/CRNN/seedv_independent_train_val_test_session1_lr1e-4.log
#ALLRound Mean and Std of acc : 0.3940/0.0969
#ALLRound Mean and Std of macro-f1 : 0.3893/0.0668

#deapv depend
#CUDA_VISIBLE_DEVICES=3 nohup python CRNN_train.py -model CRNN -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 0.0001 >/data1/cxx/CRNN/seed_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5936/0.0984
#ALLRound Mean and Std of macro-f1 : 0.5312/0.1149

#deapv indep
#CUDA_VISIBLE_DEVICES=0 nohup python CRNN_train.py -model CRNN -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/CRNN/deapv_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.4528/0.0479
#ALLRound Mean and Std of macro-f1 : 0.3281/0.0464

#deapa depend
#CUDA_VISIBLE_DEVICES=2 nohup python CRNN_train.py -model CRNN -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/CRNN/deapa_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5843/0.1800
#ALLRound Mean and Std of macro-f1 : 0.4980/0.1618

#deapa indep
#CUDA_VISIBLE_DEVICES=0 nohup python CRNN_train.py -model CRNN -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/CRNN/deapa_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5339/0.0990
#ALLRound Mean and Std of macro-f1 : 0.3691/0.0514

def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setup_seed(args.seed)

    # 设置设备
    device = torch.device(args.device)
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


                if len(val_eeg) == 0:
                    val_eeg = test_eeg
                    val_bio = test_bio
                    val_label = test_label
                
                if setting.dataset.startswith(('seed','mped')):
                    grid_size = (9,9)
                    train_eeg = map_channels_to_grid(train_eeg, SEED_CHANNEL_NAME, SEED_2D_GRID_LOC, grid_size)
                    val_eeg = map_channels_to_grid(val_eeg, SEED_CHANNEL_NAME, SEED_2D_GRID_LOC, grid_size)
                    test_eeg = map_channels_to_grid(test_eeg, SEED_CHANNEL_NAME, SEED_2D_GRID_LOC, grid_size)
                elif setting.dataset.startswith(('hci', 'deap')):
                    grid_size = (9,9)
                    train_eeg = map_channels_to_grid(train_eeg, DEAP_CHANNEL_NAME, DEAP_2D_GRID_LOC, grid_size)
                    val_eeg = map_channels_to_grid(val_eeg, DEAP_CHANNEL_NAME, DEAP_2D_GRID_LOC, grid_size)
                    test_eeg = map_channels_to_grid(test_eeg, DEAP_CHANNEL_NAME, DEAP_2D_GRID_LOC, grid_size)

                eeg_input_dim = eeg_channels * eeg_feature_dim
                bio_input_dim = bio_channels * bio_feature_dim
                dropout_rate=0.5
                if not setting.dataset.startswith(('seediv', 'seedv')):
                    lambda_c = 0.5
                else:
                    lambda_c = 0.1
                lambda_v = 0.1               

                print(f'eeg_input_dim :{eeg_input_dim}')
                print(f'bio_input_dim :{bio_input_dim}')

                print(f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')#EEG: 62 *5 PPS:1*33
                print(f'test_eeg_data shape:{test_eeg.shape}, test_bio_data shape:{test_bio.shape}, test_label shape:{test_label.shape}')

                model = CRNN(train_eeg.shape[3], train_bio.shape[2], train_bio.shape[1], num_classes)
                pretrain_dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio))
                pretrain_dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio))
                dataset_pretrain = torch.utils.data.ConcatDataset([pretrain_dataset_train, pretrain_dataset_val])
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg),torch.Tensor(train_bio),torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg),torch.Tensor(val_bio),torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg),torch.Tensor(test_bio),torch.Tensor(test_label))

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
                criterion = nn.CrossEntropyLoss()
                loss_func = criterion
                output_dir = make_output_dir(args, "CRNN")
                #train(model, dataset_train, dataset_val, dataset_test, device, output_dir="result/", metrics=None, metric_choose=None, optimizer=None, scheduler=None, batch_size=16, epochs=40, criterion=None)
                round_metric = train(model=model, dataset_pretrain=dataset_pretrain, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                    output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,scheduler=None, 
                                    batch_size=args.batch_size, epochs=args.epochs, criterion=criterion,test_sub_label=test_sub_label)
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

                if setting.dataset.startswith(('seed','mped')):
                    grid_size = (9,9)
                    train_eeg = map_channels_to_grid(train_eeg, SEED_CHANNEL_NAME, SEED_2D_GRID_LOC, grid_size)
                    val_eeg = map_channels_to_grid(val_eeg, SEED_CHANNEL_NAME, SEED_2D_GRID_LOC, grid_size)
                    test_eeg = map_channels_to_grid(test_eeg, SEED_CHANNEL_NAME, SEED_2D_GRID_LOC, grid_size)
                elif setting.dataset.startswith(('hci', 'deap')):
                    grid_size = (9,9)
                    train_eeg = map_channels_to_grid(train_eeg, DEAP_CHANNEL_NAME, DEAP_2D_GRID_LOC, grid_size)
                    val_eeg = map_channels_to_grid(val_eeg, DEAP_CHANNEL_NAME, DEAP_2D_GRID_LOC, grid_size)
                    test_eeg = map_channels_to_grid(test_eeg, DEAP_CHANNEL_NAME, DEAP_2D_GRID_LOC, grid_size)

                eeg_input_dim =  eeg_channels * eeg_feature_dim
                bio_input_dim = bio_channels *6
                dropout_rate = 0.5
                lambda_c = 0.5
                lambda_v = 0.1

                print(f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')#EEG: 62 *5 PPS:1*33
                print(f'test_eeg_data shape:{test_eeg.shape}, test_bio_data shape:{test_bio.shape}, test_label shape:{test_label.shape}')

                model = CRNN(train_eeg.shape[3], train_bio.shape[2], train_bio.shape[1], num_classes)
                pretrain_dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio))
                pretrain_dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio))
                dataset_pretrain = torch.utils.data.ConcatDataset([pretrain_dataset_train, pretrain_dataset_val])
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg),torch.Tensor(train_bio),torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg),torch.Tensor(val_bio),torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg),torch.Tensor(test_bio),torch.Tensor(test_label))

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
                criterion = nn.CrossEntropyLoss()
                loss_func = criterion
                output_dir = make_output_dir(args, "CRNN")
                #train(model, dataset_train, dataset_val, dataset_test, device, output_dir="result/", metrics=None, metric_choose=None, optimizer=None, scheduler=None, batch_size=16, epochs=40, criterion=None)
                round_metric = train(model=model, dataset_pretrain=dataset_pretrain, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                    output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,scheduler=None, 
                                    batch_size=args.batch_size, epochs=args.epochs, criterion=criterion,test_sub_label=test_sub_label)
                best_metrics.append(round_metric)
                if setting.experiment_mode =='sub_dependent':
                    subjects_metrics[rridx].append(round_metric)
                    
        if setting.experiment_mode == "sub_dependent":
            sub_result_log(args, subjects_metrics)
        else:
            result_log(args, best_metrics)

if __name__ == '__main__':
    # 解析命令行参数
    args = get_args_parser()
    args = args.parse_args()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 开始训练
    main(args)