from functools import cached_property
from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset
from .dataset_utils import load_raw_dataset_from_file, get_raw_dataset_loader
import os.path as osp
import pdb
logger = getLogger(__name__)


class CMExam(MultipleChoiceDataset):
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

    instruction = "以下是关于(中国医学考试)的单项选择题，直接给出正确答案的选项”。\n\n题目：{{question}}{{'\n' + options if options}}\n答案："
    evaluation_set = "test"
    example_set = "val"
    # load_args = ("/data/home/xiangxu_zhang/codes_repo/HealthLLM/benchmark/CMExam/data",)

    # def format_instance(self, instance):
    #     instance["target_idx"] = ord(instance["Answer"]) - ord('A')
    #     instance["options"] = [instance[op] for op in ("A", "B", "C", "D")]
    #     return instance
    
    def format_instance(self, instance):
        options = {item.split(' ')[0]: item.split(' ')[1] for item in instance['Options'].split('\n')}
        # print(options)
        # print(options.keys()) 
        res_options = [options[label] for label in options.keys()]
        # print(res_options)
        # print(instance["Question"])
        return dict(
            question=instance["Question"],
            target_idx=ord(instance["Answer"]) - ord('A'),
            options=res_options,
        )

    def calculate_metric(self, predictions):# 选项结果[2, 2, 4, 2, 4, 4, 4, 2, 1, 2]
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @cached_property
    def references(self):
        # print([ord(instance["Answer"]) - ord('A') for instance in self.evaluation_data])
        return [ord(instance["Answer"]) - ord('A') for instance in self.evaluation_data]
    
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        # print(dataset_path, subset_name, evaluation_set, example_set)

        example_path = osp.join(dataset_path, example_set + '.csv')
        evaluation_path = osp.join(dataset_path, evaluation_set + '.csv')
        self.evaluation_data = load_raw_dataset_from_file(evaluation_path)
        print(self.evaluation_data[0])
        # 由于使用ppl模式，只保留单选题
        self.evaluation_data = [q for q in self.evaluation_data if q['Answer'] and len(q['Answer']) == 1][:10]
        self.example_data = load_raw_dataset_from_file(example_path)
        self.example_data = [q for q in self.example_data if q['Answer'] and len(q['Answer']) == 1]
