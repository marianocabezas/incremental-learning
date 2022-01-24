import seaborn as sn
import os
import re
import nibabel as nib
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt


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
