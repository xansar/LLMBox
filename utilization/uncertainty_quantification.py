from typing import Any, Callable, Dict, List, Optional
import numpy as np
from scipy.stats import entropy


class SelfConsistency: 
    def _calculate_degree_confidence(self, prediction: List[int]) -> float:
        """
        参考https://github.com/zlin7/UQ-NLG
        Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models

        利用response的多样性来计算置信度
        Conf(x, s_j) = D_jj / m, s_j是主生成内容, D_jj等于s_j和其他生成内容的相似度之和, m是生成内容的数量

        Args:
            prediction: List[int], 模型的预测结果
        
        Returns:
            float, 置信度
        """
        # 假设第一个答案就是主要答案
        # TODO(xansar): 可以考虑使用其他答案作为主要答案
        primary_answer = prediction[0]

        # 计算主要答案与其他答案的相似度之和
        similarity_sum = sum([1 for answer in prediction if answer == primary_answer])

        # 计算置信度
        return similarity_sum / len(prediction)
        

    def __call__(self, predictions: List[List[Any]]) -> List[float]:
        r"""The function to calculate the self-consistency entropy.
        It calculates the entropy of the predictions of each instance.
        The self-consistency entropy is the average entropy of the predictions.
        """
        return [self._calculate_degree_confidence(prediction) for prediction in predictions]
    