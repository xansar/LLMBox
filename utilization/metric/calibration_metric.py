import numpy as np

from .metric import Metric
from torchmetrics.functional.classification import binary_calibration_error
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import torch
    
class ECE(Metric):
    def __init__(self, num_bins=10):
        self.num_bins = num_bins

    def __call__(self, predictions, references):
        # conf = torch.tensor(predictions)
        y_true = torch.tensor(references)
        num_bins = self.num_bins

        # # 计算-1出现的比例
        # mask = conf != -1
        # proportion_neg_one = (~mask).float().mean().item()

        # # 当-1被视作4时的计算
        # conf_raw = conf.clone()
        # conf_raw[conf == -1] = 4
        # bin_conf_raw = (torch.arange(num_bins) + 0.5) / num_bins
        y_pred_raw = torch.tensor(predictions)
        ece_raw = binary_calibration_error(
            y_pred_raw, y_true, n_bins=num_bins, norm='l1').item()
        # mce_raw = binary_calibration_error(
        #     y_pred_raw, y_true, n_bins=num_bins, norm='max').item()
        # rmsce_raw = binary_calibration_error(
        #     y_pred_raw, y_true, n_bins=num_bins, norm='l2').item()

        # 当-1的项被剔除后的计算
        # conf_matched = conf[mask]
        # y_true_matched = y_true[mask]
        # bin_conf_matched = (torch.arange(num_bins) + 0.5) / num_bins
        # y_pred_matched = bin_conf_matched[conf_matched]
        # ece_matched = binary_calibration_error(
        #     y_pred_matched, y_true_matched, n_bins=num_bins, norm='l1').item()
        # mce_matched = binary_calibration_error(
        #     y_pred_matched, y_true_matched, n_bins=num_bins, norm='max').item()
        # rmsce_matched = binary_calibration_error(
        #     y_pred_matched, y_true_matched, n_bins=num_bins, norm='l2').item()
        
        self.last_score_lists = {
            'Confidence': predictions, 
            'Accuracy': references
            }
        
        return {
                "ECE": ece_raw * 100,
                # "MCE": mce_raw,
                # "RMSCE": rmsce_raw,
            }
        
        # if proportion_neg_one == 0:
        #     return {
        #         'ECE': ece_matched,
        #         'MCE': mce_matched,
        #         'RMSCE': rmsce_matched,
        #         'Match Rate': 1 - proportion_neg_one
        #     }
        # else:
        #     return {
        #         "ECE": ece_raw,
        #         "MCE": mce_raw,
        #         "RMSCE": rmsce_raw,
        #         "ECE Matched": ece_matched,
        #         "MCE Matched": mce_matched,
        #         "RMSCE Matched": rmsce_matched,
        #         "Match Rate": 1 - proportion_neg_one
        #     }



class AUROC(Metric):
    def __call__(self, predictions, references):
        y_true = np.array(references)
        y_confs = np.array(predictions)
        roc_auc = roc_auc_score(y_true, y_confs)
        
        self.last_score_lists = {
            'Confidence': y_confs, 
            'Accuracy': y_true
            }
        
        return {
                'AUROC': roc_auc * 100
            }
    
class AUPRC(Metric):
    def __call__(self, predictions, references):
        y_true = np.array(references)
        y_confs = np.array(predictions)
        auprc_p = average_precision_score(y_true, y_confs)
        auprc_n = average_precision_score(1 - y_true, 1 - y_confs)
        
        self.last_score_lists = {
            'Confidence': y_confs, 
            'Accuracy': y_true
            }
        
        return {
                'AUPRC_P': auprc_p * 100,
                'AUPRC_N': auprc_n * 100
            }