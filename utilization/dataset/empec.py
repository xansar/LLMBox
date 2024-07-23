from functools import cached_property
from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset
from .dataset_utils import load_raw_dataset_from_file, get_raw_dataset_loader
import os.path as osp
import re
logger = getLogger(__name__)

BUG_EXAMPLE_LST = [
    "何型布氏杆菌对人的临床症状最严重？",
    "Risedronate 结构中，含有下列何种杂环？",
    "高剂量率（HDR）近接治疗的剂量率定义范围为多少",
]

class EMPEC(MultipleChoiceDataset):
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

    instruction = "以下是关于中国医学考试的单项选择题，你需要在[4]个选项中选出一个正确答案，直接给出正确答案的选项。\n\n题目：{{question}}{{'\n' + options if options}}\n答案："
    evaluation_set = "test"
    example_set = "dev"
    options_num = 4
    # load_args = ("/data/home/xiangxu_zhang/codes_repo/HealthLLM/benchmark/CMExam/data",)

    # def format_instance(self, instance):
    #     instance["target_idx"] = ord(instance["Answer"]) - ord('A')
    #     instance["options"] = [instance[op] for op in ("A", "B", "C", "D")]
    #     return instance
    
    def format_instance(self, instance):
        # question = instance["question_simp"].split('\n')[0]
        # pattern = re.compile(r"(?<=\n|\s)([A-D])\.\s*(.*?)(?=(?:\s*(?:\n|\s)[A-D]\.\s*|\Z))", re.DOTALL)
        # options = pattern.findall(instance["question_simp"].replace(question, ''))
        # options = [o[1].strip().strip('\n') for o in options] 
        # if len(options) != self.options_num:
        #     raise ValueError(f"选项数量不等于4，当前选项数量为{len(options)}")
        # options = [o for o in instance["question_simp"].split('\n')[1:]]
        # 去掉option里的选项字母
        # options = [o[2:].strip() for o in options]
        # print(options)
        # print(options.keys()) 
        # res_options = [options[label] for label in options.keys()]
        # print(res_options)
        # print(instance["Question"])
        return dict(
            question=instance['question'],
            target_idx=ord(instance["answer"]) - ord('A'),
            options=instance['options'],
        )

    def calculate_metric(self, predictions):# 选项结果[2, 2, 4, 2, 4, 4, 4, 2, 1, 2]
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @cached_property
    def references(self):
        # print([ord(instance["Answer"]) - ord('A') for instance in self.evaluation_data])
        return [ord(instance["answer"]) - ord('A') for instance in self.evaluation_data]
    

    def preprocess_dataset(self, raw_dataset):
        preprocessed_dataset = []
        for instance in raw_dataset:
            if any(substr in instance["question_simp"] for substr in BUG_EXAMPLE_LST):
                continue

            question = instance["question_simp"].split('\n')[0]
            pattern = re.compile(r"(?<=\n|\s)([A-D])\.\s*(.*?)(?=(?:\s*(?:\n|\s)[A-D]\.\s*|\Z))", re.DOTALL)
            options = pattern.findall(instance["question_simp"].replace(question, ''))
            options = [o[1].strip().strip('\n') for o in options] 
            if len(options) != self.options_num:
                raise ValueError(f"选项数量不等于4，当前选项数量为{len(options)}")
            instance["question"] = question.strip()
            instance["options"] = options
            preprocessed_dataset.append(instance)
        return preprocessed_dataset
    
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        # print(dataset_path, subset_name, evaluation_set, example_set)

        example_path = osp.join(dataset_path, example_set + '.jsonl')
        evaluation_path = osp.join(dataset_path, evaluation_set + '_8k.jsonl')
        self.evaluation_data = load_raw_dataset_from_file(evaluation_path)
        self.evaluation_data = self.preprocess_dataset(self.evaluation_data)
        self.example_data = load_raw_dataset_from_file(example_path)
        self.example_data = self.preprocess_dataset(self.example_data)
