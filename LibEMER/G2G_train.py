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
from models.G2G import CE_Label_Smooth_Loss, split_eye_data
from Trainer.G2GTraining import train



def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setup_seed(args.seed)
    # 如果要提取频域特征，需要手动设置提取的频段
    '''setting.extract_bands = [[4,8],[8,10],[8,12],[12,30],[30,45]]
    setting.eog_bands = [[4,8],[8,14],[14,31],[31,45]]
    setting.emg_bands = [[4,8],[8,14],[14,31],[31,45]]
    setting.gsr_bands = [[0, 0.6],[0.6, 1.2],[1.2, 1.8],[1.8, 2.4]]
    setting.bvp_bands = [[0,0.1],[0.1,0.2],[0.2,0.3],[0.3,0.4]]
    setting.resp_bands = [[0,0.6],[0.6,1.2],[1.2,1.8],[1.8,2.4]]
    setting.temp_bands =[[0, 0.05],[0.05, 0.1],[0.1, 0.15],[0.15, 0.2]]'''

    device = torch.device(args.device)
    assert setting.use_multimodal == True, 'You do not use multimodal data, please set use_multimodal to True'

    if setting.dataset.startswith('seed'):
        all_eeg, all_bio, all_label, eeg_channels, bio_channels, eeg_feature_dim, bio_feature_dim, num_classes = get_data(
            setting)
        eeg_data, bio_data, label = merge_to_part_multimodal(all_eeg, all_bio, all_label, setting)
        # bio_data = split_eye_data(bio_data, 6)
        # print(len(eeg_data))
        best_metrics = []
        subjects_metrics = [[] for _ in range(len(eeg_data))]
        # print(len(subjects_metrics))

        for rridx, (eeg_data_i, bio_data_i, label_i) in enumerate(zip(eeg_data, bio_data, label)):
            tts = get_split_index(eeg_data_i, label_i, setting)

            for ridx, (train_indexes, test_indexes, val_indexes) in enumerate( \
                    zip(tts['train'], tts['test'], tts['val'])):
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
                print(test_sub_label)

                train_eeg, train_bio, train_label, val_eeg, val_bio, val_label, test_eeg, test_bio, test_label = \
                    index_to_data_multimodal(eeg_data_i, bio_data_i, label_i, train_indexes, test_indexes, val_indexes,
                                             keep_dim=args.keep_dim)
                print(
                    f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')  # EEG: 62 *5 PPS:1*33
                print(
                    f'test_eeg_data shape:{test_eeg.shape}, test_bio_data shape:{test_bio.shape}, test_label shape:{test_label.shape}')

                if len(val_eeg) == 0:
                    val_eeg = test_eeg
                    val_bio = test_bio
                    val_label = test_label


                model = Model['G2G'](args)
                train_data = np.concatenate([
                    train_eeg.reshape(train_eeg.shape[0], -1),
                    train_bio.reshape(train_bio.shape[0], -1)
                ], axis=1)
                train_data = split_eye_data(train_data, 6)
                val_data = np.concatenate([
                    val_eeg.reshape(val_eeg.shape[0], -1),
                    val_bio.reshape(val_bio.shape[0], -1)
                ], axis=1)
                val_data = split_eye_data(val_data, 6)
                test_data = np.concatenate([
                    test_eeg.reshape(test_eeg.shape[0], -1),
                    test_bio.reshape(test_bio.shape[0], -1)
                ], axis=1)
                test_data = split_eye_data(test_data, 6)

                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

                g2g_params, backbone_params, fc_params = [], [], []
                for pname, p in model.named_parameters():
                    if "relation" in str(pname):
                        g2g_params += [p]
                    elif "backbone" in str(pname):
                        backbone_params += [p]
                    else:
                        fc_params += [p]

                optimizer = optim.AdamW([
                    {'params': g2g_params, 'lr': args.lr / 1.0},
                    {'params': backbone_params, 'lr': args.lr / 1.0},
                    {'params': fc_params, 'lr': args.lr / 1.0},
                ], betas=(0.9, 0.999), weight_decay=5e-4)

                criterion = nn.CrossEntropyLoss()

                output_dir = make_output_dir(args, 'G2G')
                round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val,
                                     dataset_test=dataset_test, device=args.device,
                                     optimizer=optimizer, output_dir=output_dir, metrics=args.metrics,
                                     metric_choose=args.metric_choose, batch_size=args.batch_size, epochs=args.epochs,
                                     criterion=criterion, test_sub_label=test_sub_label)

                best_metrics.append(round_metric)
                if setting.experiment_mode == 'sub_dependent':
                    subjects_metrics[rridx].append(round_metric)

        if setting.experiment_mode == "sub_dependent":
            sub_result_log(args, subjects_metrics)
        else:
            result_log(args, best_metrics)

    elif setting.dataset.startswith('deap'):
        eeg_data, bio_data, label, eeg_channels, bio_channels, eeg_feature_dim, bio_feature_dim, num_classes = get_data(
            setting)

        # 对时域信息求最大值，最小值，均值，方差，标准差，平方和
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
                    sq_sum = (trial ** 2).sum(axis=-1)

                    var = np.var(trial, axis=-1)
                    std = np.sqrt(var)
                    new_trial = np.stack([min, max, mean, std, var, sq_sum], axis=-1)
                    new_subject.append(new_trial)
                new_session.append(new_subject)
            new_bio_data.append(new_session)

        eeg_data, bio_data, label = merge_to_part_multimodal(eeg_data, new_bio_data, label, setting)

        best_metrics = []
        subjects_metrics = [[] for _ in range(len(eeg_data))]
        for rridx, (eeg_data_i, bio_data_i, label_i) in enumerate(zip(eeg_data, bio_data, label)):
            tts = get_split_index(eeg_data_i, label_i, setting)
            for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(
                    zip(tts['train'], tts['test'], tts['val'])):
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


                train_eeg, train_bio, train_label, val_eeg, val_bio, val_label, test_eeg, test_bio, test_label = \
                    index_to_data_multimodal(eeg_data_i, bio_data_i, label_i, train_indexes, test_indexes, val_indexes,
                                             keep_dim=args.keep_dim)
                print(
                    f'train_eeg_data shape:{train_eeg.shape}, train_bio_data shape:{train_bio.shape}, train_label shape:{train_label.shape}')  # EEG:32 *5, pps:8*6
                print(
                    f'test_eeg_data shape:{test_eeg.shape}, test_bio_data shape:{test_bio.shape}, test_label shape:{test_label.shape}')

                if len(val_eeg) == 0:
                    val_eeg = test_eeg
                    val_bio = test_bio
                    val_label = test_label

                model = Model['G2G'](args)
                train_data = np.concatenate([
                    train_eeg.reshape(train_eeg.shape[0], -1),
                    train_bio.reshape(train_bio.shape[0], -1)
                ], axis=1)
                train_data = split_eye_data(train_data, 8)
                val_data = np.concatenate([
                    val_eeg.reshape(val_eeg.shape[0], -1),
                    val_bio.reshape(val_bio.shape[0], -1)
                ], axis=1)
                val_data = split_eye_data(val_data, 8)
                test_data = np.concatenate([
                    test_eeg.reshape(test_eeg.shape[0], -1),
                    test_bio.reshape(test_bio.shape[0], -1)
                ], axis=1)
                test_data = split_eye_data(test_data, 8)

                dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
                dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
                dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
                g2g_params, backbone_params, fc_params = [], [], []
                for pname, p in model.named_parameters():
                    if "relation" in str(pname):
                        g2g_params += [p]
                    elif "backbone" in str(pname):
                        backbone_params += [p]
                    else:
                        fc_params += [p]

                optimizer = optim.AdamW([
                    {'params': g2g_params, 'lr': args.lr / 1.0},
                    {'params': backbone_params, 'lr': args.lr / 1.0},
                    {'params': fc_params, 'lr': args.lr / 1.0},
                ], betas=(0.9, 0.999), weight_decay=5e-4)

                criterion = nn.CrossEntropyLoss()

                output_dir = make_output_dir(args, 'G2G')
                round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val,
                                     dataset_test=dataset_test, device=args.device,
                                     optimizer=optimizer, output_dir=output_dir, metrics=args.metrics,
                                     metric_choose=args.metric_choose, batch_size=args.batch_size, epochs=args.epochs,
                                     criterion=criterion, test_sub_label=test_sub_label)

                best_metrics.append(round_metric)
                if setting.experiment_mode == 'sub_dependent':
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