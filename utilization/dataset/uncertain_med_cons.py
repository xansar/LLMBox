from functools import cached_property
from logging import getLogger
import random
from typing import List, Tuple

from .multiple_choice_dataset import MultipleChoiceDataset
from .dataset_utils import load_raw_dataset_from_file, get_raw_dataset_loader
import os.path as osp
import re
import string
from ..metric import ECE

logger = getLogger(__name__)


class UncertainMedCons(MultipleChoiceDataset):
    """The dataset of CMMLU.

    CMMLU: Measuring massive multitask language understanding in Chinese by Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin.

    Example:
        "Question": "在农业生产中被当作极其重要的劳动对象发挥作用，最主要的不可替代的基本生产资料是",
        "A": "农业生产工具",
        "B": "土地",
        "C": "劳动力",
        "D": "资金",
        "Answer": "B"
    """

    instruction = "现在需要评估下面这个问题的答案。\n问题：\"{{question}}\"\n现在有两个候选答案：\n1号候选答案：\n{{answer_1}}\n\n2号候选答案：\n{{answer_2}}\n1号候选答案是否在语义上蕴含了2号候选答案？请从下列选项中选择一项回答。\n{{'\n' + options if options}}\n\n回答："
    evaluation_set = "test"
    metrics = [ECE(num_bins=10)]
    example_set = "val"
    # load_args = ("/data/home/xiangxu_zhang/codes_repo/HealthLLM/benchmark/CMExam/data",)

    # def format_instance(self, instance):
    #     instance["target_idx"] = ord(instance["Answer"]) - ord('A')
    #     instance["options"] = [instance[op] for op in ("A", "B", "C", "D")]
    #     return instance
    
    def format_instance(self, instance):
        question = instance['question']
        answer_1, answer_2 = instance['processed_prediction']
        return dict(
            question=question,
            answer_1=answer_1,
            answer_2=answer_2,
            target_idx=0,
            options=['是', '否'],
        )

    def calculate_metric(self, predictions):# 选项结果[2, 2, 4, 2, 4, 4, 4, 2, 1, 2]
        # 计算 ECE

        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @cached_property
    def references(self):
        # print([ord(instance["Answer"]) - ord('A') for instance in self.evaluation_data])
        return [0 for instance in self.evaluation_data]
    
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        # print(dataset_path, subset_name, evaluation_set, example_set)

        evaluation_path = osp.join(dataset_path)
        raw_results = load_raw_dataset_from_file(evaluation_path)
        # 一个instance中的processed_prediction包含多个回复结果
        # 需要构建任一对结果之间的蕴含测试
        
        self.evaluation_data = []
        for idx, instance in enumerate(raw_results):
            processed_predictions = instance['processed_prediction']
            # 对于processed_predictions中任意两个Prediction，构建一个pair
            for i in range(len(processed_predictions)):
                for j in range(i + 1, len(processed_predictions)):
                    self.evaluation_data.append({
                        'question': instance['question'],
                        'processed_prediction': [processed_predictions[i], processed_predictions[j]],
                        'metric': instance['metric']
                    })
                    self.evaluation_data.append({
                        'question': instance['question'],
                        'processed_prediction': [processed_predictions[j], processed_predictions[i]],
                        'metric': instance['metric']
                    })

            if idx == 10:
                break
        