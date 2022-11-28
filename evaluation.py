import seaborn as sn
import os
import re
import nibabel as nib
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt


"""
Segmentation metrics
"""

def dsc_f(data_dict):
    metric_list = [
        2 * tpv / (2 * tpv + fnv + fpv)
        for tpv, fnv, fpv in zip(
            data_dict['TPV'],
            data_dict['FNV'],
            data_dict['FPV']
        )
    ]
    return metric_list


def vtpf_f(data_dict):
    metric_list = [
        tpv / (tpv + fnv)
        for tpv, fnv in zip(
            data_dict['TPV'],
            data_dict['FNV']
        )
    ]
    return metric_list


def vfpf_f(data_dict):
    metric_list = [
        fpv / (tpv + fpv) if (tpv + fpv) > 0 else 0
        for fpv, tpv in zip(
            data_dict['FPV'],
            data_dict['TPV']
        )
    ]
    return metric_list


def vppv_f(data_dict):
    metric_list = [
        1 - fpv / (tpv + fpv) if (tpv + fpv) > 0 else 0
        for fpv, tpv in zip(
            data_dict['FPV'],
            data_dict['TPV']
        )
    ]
    return metric_list


def dtpf_f(data_dict):
    metric_list = [
        tpr / gtr
        for tpr, gtr in zip(
            data_dict['TPR'],
            data_dict['GTR']
        )
    ]
    return metric_list


def dfpf_f(data_dict):
    metric_list = [
        fpr / r if r > 0 else 0
        for fpr, r in zip(
            data_dict['FPR'],
            data_dict['R']
        )
    ]
    return metric_list


def dppv_f(data_dict):
    metric_list = [
        1 - fpr / r if r > 0 else 0
        for fpr, r in zip(
            data_dict['FPR'],
            data_dict['R']
        )
    ]
    return metric_list


def f1_f(data_dict):
    metric_list = [
        2 * tpr / (gtr + r)
        for tpr, gtr, r in zip(
            data_dict['TPR'],
            data_dict['GTR'],
            data_dict['R']
        )
    ]
    return metric_list


"""
Classification metrics
"""


def acc_f(data_dict):
    metric_list = [
        (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        for tp, tn, fp, fn in zip(
            data_dict['TP'],
            data_dict['TN'],
            data_dict['FP'],
            data_dict['FN']
        )
    ]
    return metric_list


def bacc_f(data_dict):
    t0_list = t0r_f(data_dict)
    t1_list = t1r_f(data_dict)
    return np.mean([t0_list, t1_list], axis=0).tolist()


def t0r_f(data_dict):
    metric_list = [
        tn / (tn + fp) if (tn + fp) > 0 else 0
        for tn, fp in zip(
            data_dict['TN'],
            data_dict['FP']
        )
    ]
    return metric_list


def t1r_f(data_dict):
    metric_list = [
        tp / (tp + fn) if (tp + fn) > 0 else 0
        for tp, fn in zip(
            data_dict['TP'],
            data_dict['FN']
        )
    ]
    return metric_list


"""
Memory metrics
"""


def fwt(metrics_grid):
    if len(metrics_grid.shape) == 2:
        if metrics_grid.shape[0] == metrics_grid.shape[1] + 2:
            m_previous = np.diag(metrics_grid[2:-1, 1:])
            m_init = metrics_grid[1, 1:]
            return np.mean(m_previous - m_init)
        else:
            metrics_avg = np.mean(metrics_grid, axis=1)
            m_previous = metrics_avg[2:-1]
            m_init = metrics_avg[1]
    else:
        m_previous = metrics_grid[2:-1]
        m_init = metrics_grid[1]
    return np.mean(m_previous - m_init)


def bwt(metrics_grid):
    if len(metrics_grid.shape) == 2:
        if metrics_grid.shape[0] == metrics_grid.shape[1] + 2:
            m_task = np.diag(metrics_grid[2:-1, :-1])
            m_last = metrics_grid[-1, :-1]
        else:
            metrics_avg = np.mean(metrics_grid, axis=1)
            m_task = metrics_avg[2:-1]
            m_last = metrics_avg[-1]
    else:
        m_task = metrics_grid[2:-1]
        m_last = metrics_grid[-1]
    return np.mean(m_last - m_task)


def forgetting(metrics_grid):
    if len(metrics_grid.shape) == 2:
        if metrics_grid.shape[0] == metrics_grid.shape[1] + 2:
            f_t = [
                np.max(
                    metrics_grid[2:t + 3, :(t + 1)] - metrics_grid[t + 3, :(t + 1)],
                    axis=0
                )
                for t in range(len(metrics_grid) - 3)
            ]
            F_t = [np.mean(f_ti) for f_ti in f_t]
        else:
            metrics_vec = np.mean(metrics_grid, axis=1)
            F_t = np.array([
                np.max(metrics_vec[2:t + 3]) - metrics_vec[t + 3]
                for t in range(len(metrics_grid) - 3)
            ])
    else:
        F_t = np.array([
            np.max(metrics_grid[2:t + 3]) - metrics_grid[t + 3]
            for t in range(len(metrics_grid) - 3)
        ])
    return F_t


def intransigence(metrics_grid):
    if len(metrics_grid.shape) == 2:
        if metrics_grid.shape[0] == metrics_grid.shape[1] + 2:
            I_i = metrics_grid[0, :] - np.diag(metrics_grid[2:, :])
        else:
            I_i = metrics_grid[0, :] - np.max(metrics_grid[2:, :], axis=0)
    else:
        I_i = metrics_grid[0] - np.max(metrics_grid[2:])
    return I_i


def transfer(metrics_grid):
    if len(metrics_grid.shape) == 2:
        if metrics_grid.shape[0] == metrics_grid.shape[1] + 2:
            tr = bwt(metrics_grid) + fwt(metrics_grid)
        else:
            metrics_vec = np.mean(metrics_grid, axis=1)
            tr = np.mean([
                metrics_vec[t] - metrics_vec[t-1]
                for t in range(2, len(metrics_grid))
            ])
    else:
        tr = np.mean([
            metrics_grid[t] - metrics_grid[t-1]
            for t in range(2, len(metrics_grid))
        ])
    return tr


def learning(metrics_grid):
    if len(metrics_grid.shape) == 2:
        if metrics_grid.shape[0] == metrics_grid.shape[1] + 2:
            tr = np.mean(np.diag(metrics_grid[2:, :]))
        else:
            tr = transfer(metrics_grid)
    else:
        tr = np.mean(metrics_grid)
    return tr
