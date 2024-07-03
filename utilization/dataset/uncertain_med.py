from functools import cached_property
from logging import getLogger
import random

from .multiple_choice_dataset import MultipleChoiceDataset
from .dataset_utils import load_raw_dataset_from_file, get_raw_dataset_loader
import os.path as osp
import re
from ..metric import ECE

logger = getLogger(__name__)


class UncertainMed(MultipleChoiceDataset):
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

    instruction = "以下是关于中国医学考试的单项选择题以及一个答案。\n\n题目：{{question}}\n答案：{{raw_prediction}}\n请评估上述答案正确的可能性，并从下列可能性选项中选择一项：\n{{'\n' + options if options}}\n仅回复可能性选项，不要回复其他内容。\n可能性："
    evaluation_set = "test"
    metrics = [ECE(num_bins=10)]
    example_set = "val"
    # load_args = ("/data/home/xiangxu_zhang/codes_repo/HealthLLM/benchmark/CMExam/data",)

    # def format_instance(self, instance):
    #     instance["target_idx"] = ord(instance["Answer"]) - ord('A')
    #     instance["options"] = [instance[op] for op in ("A", "B", "C", "D")]
    #     return instance
    
    def format_instance(self, instance):
        # question
        pattern = re.compile(r'题目：(.*?)答案：', re.DOTALL)
        match = pattern.search(instance["source"])
        # 提取匹配到的内容
        if match:
            question_content = match.group(1).strip()
            # 将字母序号换为数字序号
            for c in 'ABCDE':
                question_content = question_content.replace('\n' + c, '\n' + str(ord(c) - ord('A') + 1))
        else:
            raise ValueError("未找到匹配的内容")
        question_content += '\n'
        
        # 调整预测,如果模型没有正确回答选项（为-1），则直接随机
        if 'prediction' in instance:
            raw_prediction = instance['prediction'] + 1
        elif 'processed_prediction' in instance:
            raw_prediction = instance['processed_prediction'][0] + 1
        else:
            raise ValueError("未找到预测结果")
        
        if raw_prediction == -1:
            raw_prediction = random.randint(1, 5)

        pattern = fr"{raw_prediction}\. .+?"
        match = re.search(pattern, question_content)
        
        if match:
            prediction_content = match.group(0)
        else:
            raise ValueError("未找到匹配的预测")
        
        # 调整选项
        options = [f"{i}-{i+10}\%" for i in range(0, 100, 10)]
        return dict(
            question=question_content,
            raw_prediction=prediction_content,
            target_idx=0,
            options=options,
        )


    def calculate_metric(self, predictions):# 选项结果[2, 2, 4, 2, 4, 4, 4, 2, 1, 2]
        # 计算 ECE
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @cached_property
    def references(self):
        # print([ord(instance["Answer"]) - ord('A') for instance in self.evaluation_data])
        return [int(instance['metric']['Accuracy']) for instance in self.evaluation_data]
    
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        # print(dataset_path, subset_name, evaluation_set, example_set)

        evaluation_path = osp.join(dataset_path)
        self.evaluation_data = load_raw_dataset_from_file(evaluation_path)
        print(self.evaluation_data[0])
        # 由于使用ppl模式，只保留单选题
        # self.evaluation_data = [q for q in self.evaluation_data if q['Answer'] and len(q['Answer']) == 1][:10]
        # self.example_data = load_raw_dataset_from_file(example_path)
        # self.example_data = [q for q in self.example_data if q['Answer'] and len(q['Answer']) == 1]
