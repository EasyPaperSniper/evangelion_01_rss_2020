###============ This file is utils for plotting things==============

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re


def read_log_file(file_name, key_name, value_name, smooth=3):
    keys, values = [], []
    try:
        with open(file_name, 'r') as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    key, value = e[key_name], e[value_name]
                    keys.append(int(key))
                    values.append(float(value))
                except:
                    pass
    except:
        print('bad file: %s' % file_name)
        return None, None
    keys, values = np.array(keys), np.array(values)
    if smooth > 1 and values.shape[0] > 0:
        K = np.ones(smooth)
        ones = np.ones(values.shape[0])
        values = np.convolve(values, K, 'same') / np.convolve(ones, K, 'same')

    return keys, values


def parse_log_files(file_name_template,
                    seed_index,
                    key_name,
                    value_name,
                    best_k=None,
                    ignore_seeds=0):
    all_values = []
    progress = []
    num_keys = int(1e9)
    actual_keys = None

    file_name = file_name_template 
    for seed in seed_index:
        file_name = file_name_template % seed
        keys, values = read_log_file(file_name, key_name, value_name)
        if keys is None or keys.shape[0] == 0:
            continue
        progress.append(keys.shape[0])
        if actual_keys is None or actual_keys.shape[0] < keys.shape[0]:
            actual_keys = keys
        all_values.append(values)

    if len(all_values) == 0:
        return None, None, None

    threshold = sorted(progress)[ignore_seeds]

    means, half_stds = [], []
    for i in range(threshold):
        vals = []

        for v in all_values:
            if i < v.shape[0]:
                vals.append(v[i])
        if best_k is not None:
            vals = sorted(vals)[-best_k:]
        means.append(np.mean(vals))
        half_stds.append(0.5 * np.std(vals))
    means = np.array(means)
    half_stds = np.array(half_stds)

    actual_keys = actual_keys[:threshold]
    assert means.shape[0] == actual_keys.shape[0]

    return actual_keys, means, half_stds


def print_result(root,
                 name,
                 title,
                 label=None,
                 seed_index=1,
                 key_name='step',
                 value_name='episode_reward',
                 max_time=None,
                 best_k=None,
                 timescale=1,
                 ignore_seeds=0):
    file_name = name + '.log'
    file_name_template = os.path.join(root, 'trial_%d/' , file_name )
    keys, means, half_stds = parse_log_files(
        file_name_template,
        seed_index,
        key_name,
        value_name,
        best_k=best_k,
        ignore_seeds=ignore_seeds)
    label = label or root.split('/')[-1]
    if keys is None:
        return

    if max_time is not None:
        idxs = np.where(keys <= max_time)
        keys = keys[idxs]
        means = means[idxs]
        half_stds = half_stds[idxs]

    keys *= timescale

    plt.locator_params(nbins=6, axis='x')
    plt.locator_params(nbins=10, axis='y')
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)

    plt.grid(alpha=0.8)
    plt.title(title)
    plt.plot(keys, means, label=label)
    plt.fill_between(keys, means - half_stds, means + half_stds, alpha=0.2)
    plt.legend(
        loc='lower right', prop={
            'size': 6
        }).get_frame().set_edgecolor('0.1')
    plt.xlabel(key_name)
    plt.ylabel(value_name)


######################################################################################################

def print_target_eval(root,
                 title,
                 seed_index=1,
                 key_name='step',
                 value_name='episode_reward',
                 max_time=None,
                 best_k=None,
                 timescale=1,
                 ignore_seeds=0):
    file_name_1 = 'eval.log'
    file_name_2 = 'tgt.log'
    file_name_template_1 = os.path.join(root, 'trial_%d/' , file_name_1 )
    keys, means, half_stds = parse_log_files(
        file_name_template_1,
        seed_index,
        key_name,
        value_name,
        best_k=best_k,
        ignore_seeds=ignore_seeds)

    if keys is None:
        return
    
    file_name_template_2 = os.path.join(root, 'trial_%d/' , file_name_2 )
    tgt_keys, tgt_means, tgt_half_stds = parse_log_files(
        file_name_template_2,
        seed_index,
        key_name,
        value_name,
        best_k=best_k,
        ignore_seeds=ignore_seeds)
    
    if keys is None:
        return

    if max_time is not None:
        idxs = np.where(keys <= max_time)
        keys = keys[idxs]
        means = means[idxs]
        half_stds = half_stds[idxs]

    keys *= timescale

    plt.locator_params(nbins=6, axis='x')
    plt.locator_params(nbins=10, axis='y')
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)

    plt.grid(alpha=0.8)
    plt.title(title)
    plt.plot(keys, means,  label = 'prediction')
    plt.fill_between(keys, means - half_stds, means + half_stds, alpha=0.2)
    plt.plot(tgt_keys, tgt_means,  label = 'ground_truth')
    plt.fill_between(tgt_keys, tgt_means - tgt_half_stds, tgt_means + tgt_half_stds, alpha=0.2)
    plt.legend(
        loc='lower right', prop={
            'size': 6
        }).get_frame().set_edgecolor('0.1')
    plt.xlabel(key_name)
    plt.ylabel(value_name)

