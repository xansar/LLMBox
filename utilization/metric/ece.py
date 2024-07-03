import numpy as np

from .metric import Metric
from torchmetrics.functional.classification import binary_calibration_error
import torch


# class ECE(Metric):
#     r"""Calculate calibration error
#     """
#     def __init__(self, num_bins=10):
#         self.num_bins = num_bins

#     def __call__(self, predictions, references):
#         conf = torch.tensor(predictions)
#         # 如果conf中存在-1，说明模型没有预测，直接返回0
#         conf[conf == -1] = 4
#         y_true = torch.tensor(references)
#         num_bins = self.num_bins

#         # 计算每个bin的准确率和置信度
#         bin_conf = (torch.arange(num_bins) + 0.5) / num_bins
#         y_pred = bin_conf[conf]
#         ece = binary_calibration_error(
#             y_pred, y_true, n_bins=self.num_bins, norm='l1')
#         mce = binary_calibration_error(
#             y_pred, y_true, n_bins=self.num_bins, norm='max')
#         rmsce = binary_calibration_error(
#             y_pred, y_true, n_bins=self.num_bins, norm='l2')
    

#         self.last_score_lists = {'Confidence': conf.tolist(), 'Accuracy': y_true.tolist()}

#         return {
#             "ECE": ece.item(),
#             "MCE": mce.item(),
#             "RMSCE": rmsce.item()
#         }
    
class ECE(Metric):
    def __init__(self, num_bins=10):
        self.num_bins = num_bins

    def __call__(self, predictions, references):
        conf = torch.tensor(predictions)
        y_true = torch.tensor(references)
        num_bins = self.num_bins

        # 计算-1出现的比例
        mask = conf != -1
        proportion_neg_one = (~mask).float().mean().item()

        # 当-1被视作4时的计算
        conf_raw = conf.clone()
        conf_raw[conf == -1] = 4
        bin_conf_raw = (torch.arange(num_bins) + 0.5) / num_bins
        y_pred_raw = bin_conf_raw[conf_raw]
        ece_raw = binary_calibration_error(
            y_pred_raw, y_true, n_bins=num_bins, norm='l1').item()
        mce_raw = binary_calibration_error(
            y_pred_raw, y_true, n_bins=num_bins, norm='max').item()
        rmsce_raw = binary_calibration_error(
            y_pred_raw, y_true, n_bins=num_bins, norm='l2').item()

        # 当-1的项被剔除后的计算
        conf_matched = conf[mask]
        y_true_matched = y_true[mask]
        bin_conf_matched = (torch.arange(num_bins) + 0.5) / num_bins
        y_pred_matched = bin_conf_matched[conf_matched]
        ece_matched = binary_calibration_error(
            y_pred_matched, y_true_matched, n_bins=num_bins, norm='l1').item()
        mce_matched = binary_calibration_error(
            y_pred_matched, y_true_matched, n_bins=num_bins, norm='max').item()
        rmsce_matched = binary_calibration_error(
            y_pred_matched, y_true_matched, n_bins=num_bins, norm='l2').item()
        
        self.last_score_lists = {
            'Confidence': predictions, 
            'Accuracy': references
            }
        
        if proportion_neg_one == 0:
            return {
                'ECE': ece_matched,
                'MCE': mce_matched,
                'RMSCE': rmsce_matched,
                'Match Rate': 1 - proportion_neg_one
            }
        else:
            return {
                "ECE": ece_raw,
                "MCE": mce_raw,
                "RMSCE": rmsce_raw,
                "ECE Matched": ece_matched,
                "MCE Matched": mce_matched,
                "RMSCE Matched": rmsce_matched,
                "Match Rate": 1 - proportion_neg_one
            }