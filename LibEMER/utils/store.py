import argparse
import os.path
import time
from pathlib import Path

import torch

from config.setting import resolve_effective_experiment_mode


def make_output_dir(args, model):
    output_dir = Path(args.output_dir)
    output_dir = output_dir / model
    if args.setting is not None:
        output_dir = output_dir / args.setting
    else:
        output_dir = output_dir / args.dataset / args.experiment_mode
        output_dir = output_dir / args.split_type
    if args.label_used is not None:
        if len(args.label_used) == 1:
            output_dir = output_dir / args.label_used[0]
        else:
            output_dir = output_dir / "both".join(label for label in args.label_used)
    return output_dir


def save_state(output_dir, model, optimizer, epoch, r_idx='last', rr_idx='last', metric=None, state='best'):
    # compatibility
    if type(output_dir) is argparse.Namespace:
        output_dir = make_output_dir(output_dir, output_dir.model)
    else:
        output_dir = Path(output_dir)
    if not (r_idx == 'last' and rr_idx == 'last'):
        output_dir = output_dir / str(r_idx)
        output_dir = output_dir / str(rr_idx)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"An error occurred: {e.strerror}")
    checkpoint_path = output_dir / f'checkpoint-{str(epoch)}' if metric is None \
        else output_dir / f'checkpoint-{state}{metric}'
    save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(save, checkpoint_path)
    print(f"save model to {checkpoint_path}")


def save_data(args, data, label):
    save_dir = Path(args.data_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mode = {
        'subject-dependent': 'sub-dep',
        'subject-independent': 'sub-In',
        'cross-session': 'cro-sess'
    }
    save_path = save_dir / f'{args.dataset}'
    save_path = save_path / f'{args.feature_type}-tw-{args.time_window}ol-{args.overlap}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Saving Processed Data To {save_path}")


def _build_log_file_name(args):
    protocol_map = {
        'sub_dependent': 'SD',
        'sub_independent': 'SI',
    }
    protocol = protocol_map.get(resolve_effective_experiment_mode(args))
    if protocol is None:
        return time.strftime("%Y-%m-%d %H_%M_%S", args.time)

    name_parts = [args.model, args.dataset, protocol]
    if args.label_used:
        if len(args.label_used) == 1:
            name_parts.append(args.label_used[0])
        else:
            name_parts.append("both".join(label for label in args.label_used))
    name_parts.append(time.strftime("%Y-%m-%d", args.time))
    return "_".join(name_parts)


def save_res(args, metric):
    log_dir = Path(args.log_dir)
    add_dir(log_dir)
    log_file = log_dir / _build_log_file_name(args)
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write(str(args))
    with open(log_file, 'a') as f:
        f.write('\n' + str(metric))


def add_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
