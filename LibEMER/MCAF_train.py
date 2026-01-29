from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args

from data_utils.load_data import get_data
from data_utils.preprocess import preprocess, label_process, multimodal_preprocess, normalize
from data_utils.split import get_split_index, index_to_data_multimodal, merge_to_part_multimodal

from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from models.MCAF import MCAF
from Trainer.MCAFTraining import train

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# python MCAF_train.py -model MCAF -use_multimodal -dataset seed_de_lds -dataset_path SEED -onehot -sessions 1 2 3 -split_type leave_one_out -lr 0.01 -batch_size 32 -epochs 80
# ALLRound Mean and Std of acc : 0.8832/0.0728


#seed depend
#CUDA_VISIBLE_DEVICES=2 nohup python MCAF_train.py -model MCAF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path SEED -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/MCAF/seed_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5211/0.2477
#ALLRound Mean and Std of macro-f1 : 0.4275/0.2771

#seed indep
#CUDA_VISIBLE_DEVICES=2 nohup python MCAF_train.py -model MCAF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/SEED -dataset seed_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/MCAF/seed_independent_train_val_test_session1_lr1e-4.log
#ALLRound Mean and Std of acc : 0.2945/0.0537
#ALLRound Mean and Std of macro-f1 : 0.2272/0.0911

#seedv depend
#CUDA_VISIBLE_DEVICES=2 nohup python MCAF_train.py -model MCAF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/MCAF/seedv_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.0764/0.1639
#ALLRound Mean and Std of macro-f1 : 0.0480/0.0983

#seedv indep
#CUDA_VISIBLE_DEVICES=0 nohup python MCAF_train.py -model MCAF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path SEEDV -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot >/data1/cxx/MCAF/seedv_independent_train_val_test_session1_lr1e-4.log
#ALLRound Mean and Std of acc : 0.2271/0.0257
#ALLRound Mean and Std of macro-f1 : 0.1199/0.0326

#deapv depend
#CUDA_VISIBLE_DEVICES=3 nohup python MCAF_train.py -model MCAF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/MCAF/deapv_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5359/0.1634
#ALLRound Mean and Std of macro-f1 : 0.4642/0.1638

#deapv indep
#CUDA_VISIBLE_DEVICES=2 nohup python MCAF_train.py -model MCAF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/MCAF/deapv_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.5208/0.0749
#ALLRound Mean and Std of macro-f1 : 0.3634/0.0544

#deapa depend
#CUDA_VISIBLE_DEVICES=1 nohup python MCAF_train.py -model MCAF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/MCAF/deapa_dependent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.6229/0.1521
#ALLRound Mean and Std of macro-f1 : 0.4863/0.1409

#deapa indep
#CUDA_VISIBLE_DEVICES=2 nohup python MCAF_train.py -model MCAF -use_multimodal -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path /data1/cxx/data/data_preprocessed_python/data_preprocessed_python -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used arousal -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4 >/data1/cxx/MCAF/deapa_independent_train_val_test_lr1e-4.log
#ALLRound Mean and Std of acc : 0.4958/0.1066
#ALLRound Mean and Std of macro-f1 : 0.3585/0.0690

'''
def main(args):
    # 配置设置
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed) 
    device = torch.device(args.device)
    
    assert setting.use_multimodal == True, 'You do not use multimodal data, please set use_multimodal to True'
    
    # 加载EEG和EOG数据
    all_eeg, all_bio, all_label,eeg_channels, bio_channels, eeg_feature_dim, bio_feature_dim, num_classes = get_data(setting)
    # eeg_data：[3, 12, 15, 58, 62, 5] bio_data：[3, 12, 15, 58, 1, 33]
    # session, subject, trail, channel, original_data
    eeg_data, bio_data, label = merge_to_part_multimodal(all_eeg, all_bio, all_label,setting)
    
    # len(eeg_data) 36

    best_metrics = []
    subjects_metrics = [[]for _ in range(len(eeg_data))]
    
    # len(subjects_metrics) 36
    eeg_channels = 62
    eog_channels = 1
    seq_len_eeg = 5
    seq_len_eog = 33
    num_classes = 3
    num_channels = 32
    seq_len = 128
    d_model = 64
    nhead = 4
    num_layers = 2

    for rridx, (eeg_data_i, bio_data_i, label_i) in enumerate(zip(eeg_data, bio_data, label)):
        tts = get_split_index(eeg_data_i, label_i, setting)
            
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(\
            zip(tts['train'], tts['test'], tts['val'])):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f'train indexes:{train_indexes}, test indexes:{test_indexes}')
            else:
                print(f'train indexes:{train_indexes}, test indexes:{test_indexes}, val indexes:{val_indexes}')
            train_eeg, train_bio,train_label, val_eeg,val_bio,val_label, test_eeg,test_bio, test_label =\
                index_to_data_multimodal(eeg_data_i, bio_data_i, label_i,train_indexes,test_indexes,val_indexes,keep_dim=args.keep_dim)
            print(f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')
            print(f'test_eeg_data shape:{test_eeg.shape}, test_bio_data shape:{test_bio.shape}, test_label shape:{test_label.shape}')
            # train_eeg_data shape:(454, 6, 62, 5), train_bio_data shape:(454, 6, 1, 33), train_label shape:(454, 3)
            # test_eeg_data shape:(313, 6, 62, 5), test_bio_data shape:(313, 6, 1, 33), test_label shape:(313, 3)

            if len(val_eeg) == 0:
                val_eeg = test_eeg
                val_bio = test_bio
                val_label = test_label
                   
            model = Model['MCAF'](
                num_classes=num_classes,
                eeg_channels=eeg_channels,  # EEG通道数
                eog_channels=eog_channels,  # EOG通道数
                seq_len_eeg=seq_len_eeg,    # EEG序列长度
                seq_len_eog=seq_len_eog,    # EOG序列长度
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers
            )
            # 将标签从one-hot转为类别索引
            train_label = np.argmax(train_label, axis=-1) if train_label.ndim > 1 else train_label
            val_label = np.argmax(val_label, axis=-1) if val_label.ndim > 1 else val_label
            test_label = np.argmax(test_label, axis=-1) if test_label.ndim > 1 else test_label
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.LongTensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.LongTensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.LongTensor(test_label))

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 1e-2)
            criterion = nn.CrossEntropyLoss()

            output_dir = make_output_dir(args, 'MCAF')
            round_metric = train(model = model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                optimizer=optimizer, criterion=criterion, output_dir=output_dir, 
                                metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs) 
                
            best_metrics.append(round_metric)
            if setting.experiment_mode =='sub_dependent':
                subjects_metrics[rridx].append(round_metric)
                         
    if setting.experiment_mode == "sub_dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics) '''

def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setup_seed(args.seed)

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
                
                seq_len_eeg = train_eeg.shape[-1]
                seq_len_eog = train_bio.shape[-1]
                eeg_channels = train_eeg.shape[-2]
                eog_channels = train_bio.shape[-2]
                d_model = 64
                nhead = 4
                num_layers = 2
                eeg_dropout_rate = 0.6 #0.7
                bio_dropout_rate = 0.6 #0.7               

                model = Model['MCAF'](num_classes=num_classes, eeg_channels=eeg_channels, eog_channels=eog_channels, seq_len_eeg=seq_len_eeg, seq_len_eog=seq_len_eog, d_model=d_model, nhead=nhead, num_layers=num_layers)
                pretrain_dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio))
                pretrain_dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio))
                dataset_pretrain = torch.utils.data.ConcatDataset([pretrain_dataset_train, pretrain_dataset_val])
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 1e-2)
                criterion = nn.CrossEntropyLoss()

                output_dir = make_output_dir(args, 'MCAF')
                round_metric = train(model = model, dataset_pretrain=dataset_pretrain, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    optimizer=optimizer, criterion=criterion, output_dir=output_dir, 
                                    metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs,test_sub_label=test_sub_label) 
                
                
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

                seq_len_eeg = train_eeg.shape[-1]
                seq_len_eog = train_bio.shape[-1]
                eeg_channels = train_eeg.shape[-2]
                eog_channels = train_bio.shape[-2]
                d_model = 64
                nhead = 4
                num_layers = 2
                eeg_dropout_rate = 0.6 #0.7
                bio_dropout_rate = 0.6 #0.7               

                model = Model['MCAF'](num_classes=num_classes, eeg_channels=eeg_channels, eog_channels=eog_channels, seq_len_eeg=seq_len_eeg, seq_len_eog=seq_len_eog, d_model=d_model, nhead=nhead, num_layers=num_layers)
                pretrain_dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio))
                pretrain_dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio))
                dataset_pretrain = torch.utils.data.ConcatDataset([pretrain_dataset_train, pretrain_dataset_val])
                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg), torch.Tensor(train_bio), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg), torch.Tensor(val_bio), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg), torch.Tensor(test_bio), torch.Tensor(test_label))

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 1e-2)
                criterion = nn.CrossEntropyLoss()

                output_dir = make_output_dir(args, 'MCAF')
                round_metric = train(model = model, dataset_pretrain=dataset_pretrain, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device = args.device,
                                    optimizer=optimizer, criterion=criterion, output_dir=output_dir, 
                                    metrics = args.metrics, metric_choose=args.metric_choose,batch_size=args.batch_size, epochs = args.epochs,test_sub_label=test_sub_label) 
                
                
                best_metrics.append(round_metric)
                if setting.experiment_mode =='sub_dependent':
                    subjects_metrics[rridx].append(round_metric)
                    
        if setting.experiment_mode == "sub_dependent":
            sub_result_log(args, subjects_metrics)
        else:
            result_log(args, best_metrics) 

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    setup_seed(args.seed)
    main(args)