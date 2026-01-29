import numpy as np

from config.setting import set_setting_by_args, preset_setting

from data_utils.load_data import get_data
from data_utils.preprocess import preprocess, label_process, multimodal_preprocess
from data_utils.split import get_split_index, index_to_data_multimodal, merge_to_part_multimodal

from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from data_utils.preprocess import normalize

from Trainer.training import train
from models.HetEmotionNet import HetEmotionNet
import torch
import torch.optim as optim
import torch.nn as nn

#python test_deap_multimodal.py -dataset_path data_preprocessed_python -dataset deap -use_multimodal -sample_length 128 -stride 128 -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -only_seg -onehot

#python test_deap_multimodal.py -dataset_path data_preprocessed_python -dataset deap -use_multimodal -extract_bio -sample_length 9 -stride 1 -bio_length 9 -bio_stride 1 -bounds 5 5 -label_used valence  -onehot

#python test_deap_multimodal.py -dataset_path data_preprocessed_python -dataset deap -use_multimodal -sample_length 1 -stride 1 -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -onehot

#python test_deap_multimodal.py -dataset_path data_preprocessed_python -dataset deap -use_multimodal -sample_length 128 -stride 128 -bio_length 128 -bio_stride 128 -split_type kfold -bounds 5 5 -label_used valence -only_seg -onehot

def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setup_seed(args.seed)
    #如果要提取频域特征，需要手动设置提取的频段
    setting.extract_bands = [[4,8],[8,14],[14,31],[31,45]]
    setting.eog_bands = [[4,8],[8,14],[14,31],[31,45]]
    setting.emg_bands = [[4,8],[8,14],[14,31],[31,45]]
    setting.gsr_bands = [[0, 0.6],[0.6, 1.2],[1.2, 1.8],[1.8, 2.4]]
    setting.bvp_bands = [[0,0.1],[0.1,0.2],[0.2,0.3],[0.3,0.4]]
    setting.resp_bands = [[0,0.6],[0.6,1.2],[1.2,1.8],[1.8,2.4]]
    setting.temp_bands =[[0, 0.05],[0.05, 0.1],[0.1, 0.15],[0.15, 0.2]]

    device = torch.device(args.device)
    assert setting.use_multimodal == True, 'You do not use multimodal data, please set use_multimodal to True'

    eeg_data, bio_data, label,eeg_channels, bio_channels, eeg_feature_dim, bio_feature_dim, num_classes = get_data(setting)
    eeg_data, bio_data, label = merge_to_part_multimodal(eeg_data, bio_data, label, setting)
    device = torch.device(args.device)

    best_metrics = []
    subjects_metrics = [[] for _ in range(len(eeg_data))]
    print(len(subjects_metrics))
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
                train_eeg, train_bio, train_label, val_eeg, val_bio, val_label, test_eeg, test_bio, test_label = \
                    index_to_data_multimodal(eeg_data_i, bio_data_i, label_i, train_indexes, test_indexes,
                                             val_indexes, True)
                test_sub_num = len(test_eeg)
                test_sub_label = []
                for i in range(test_sub_num):
                    test_sub_count = len(test_eeg[i])
                    test_sub_label.extend([i + 1 for j in range(test_sub_count)])
                test_sub_label = np.array(test_sub_label)

            train_eeg_data, train_bio_data,train_label, val_eeg_data,val_bio_data,val_label, test_eeg_data,test_bio_data, test_label =\
                index_to_data_multimodal(eeg_data_i, bio_data_i, label_i,train_indexes,test_indexes,val_indexes,keep_dim=args.keep_dim)
            print(f'train_eeg_data shape:{train_eeg_data.shape}, train_bio_data shape:{train_bio_data.shape}, train_label shape:{train_label.shape}')

            if len(val_eeg_data) == 0:
                val_eeg_data = test_eeg_data
                val_bio_data = test_bio_data
                val_label = test_label

            # train_data = np.concatenate((train_eeg_data, train_bio_data), 1)
            # val_data = np.concatenate((val_eeg_data, val_bio_data), 1)
            # test_data = np.concatenate((test_eeg_data, test_bio_data), 1)

            model = HetEmotionNet(args.device, 40, 128, 4, 2)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_eeg_data), torch.Tensor(train_bio_data),torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_eeg_data), torch.Tensor(val_bio_data),torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_eeg_data), torch.Tensor(test_bio_data),torch.Tensor(test_label))
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005, eps=0.0001)
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "HetEmotionNet")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val,
                                 dataset_test=dataset_test, device=args.device,
                                 optimizer=optimizer, output_dir=output_dir, metrics=args.metrics,
                                 metric_choose=args.metric_choose, batch_size=args.batch_size, epochs=args.epochs,
                                 criterion=criterion, test_sub_label=test_sub_label)
            best_metrics.append(round_metric)
            if setting.experiment_mode == "sub_dependent":
                subjects_metrics[rridx].append(round_metric)
            # best metrics: every round metrics dict
            # subjects metrics: (subject, sub_round_metric)
    if setting.experiment_mode == "sub_dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    state_log(args)
    main(args)