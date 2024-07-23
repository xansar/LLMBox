from functools import cached_property
from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset
from .dataset_utils import load_raw_dataset_from_file, get_raw_dataset_loader
from datasets import load_dataset
import os.path as osp
import re
from ..dataset_enum import UNCERTAIN_MED_BENCH_SUBJECTS
logger = getLogger(__name__)

class UncertainMedBench(MultipleChoiceDataset):
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

    instruction_dict = {
        'empec': "请你回答下面这道关于中国医学考试的单项选择题，你需要在选项中选出一个正确答案，直接给出正确答案的选项序号。\n\n【题目】：\n\"\"\"\n{{question}}{{'\n' + options if options}}\n\"\"\"\n\n请直接写出正确答案的选项序号。",
        'cmb_exam': "请你回答下面这道关于中国医学考试的单项选择题，你需要在选项中选出一个正确答案，直接给出正确答案的选项序号。\n\n【题目】：\n\"\"\"\n{{question}}{{'\n' + options if options}}\n\"\"\"\n\n请直接写出正确答案的选项序号。",
        'dialmed': "请你回答下面这道关于问诊对话药物推荐的单项选择题，问诊对话中的关键药物被[MASK]符号掩盖，现在请你从几个候选药物中选择一个最有可能的药物填入[MASK]位置。直接给出正确答案的选项序号。\n\n【对话】：\n\"\"\"\n{{question}}\n\"\"\"\n【候选药物】{{'\n' + options if options}}\n\n请直接写出最有可能的药物的选项序号。",
        'rjua_disease': "请你回答下面这道关于疾病诊断的单项选择题，你需要基于病情陈述从几个候选诊断中选择一个最有可能的诊断，直接给出正确答案的选项序号。\n\n【病情陈述】：\n\"\"\"\n{{question}}\n\"\"\"\n【候选诊断】{{'\n' + options if options}}\n\n请直接写出最有可能的诊断的选项序号。",
        'tcmsd_tcm': "请你回答下面这道关于中医辨证的单项选择题，你需要基于病历信息（病史、主诉、检查）从几个候选证候中选择一个最有可能的证候，直接给出正确答案的选项序号。\n\n【病历信息】：\n\"\"\"\n{{question}}\n\"\"\"\n【候选证候】{{'\n' + options if options}}\n\n请直接写出最有可能的证候的选项序号。",
        'tcmsd_wm': "请你回答下面这道关于疾病诊断的单项选择题，你需要基于病历信息（病史、主诉、检查）从几个候选诊断中选择一个最有可能的诊断，直接给出正确答案的选项序号。\n\n【病历信息】：\n\"\"\"\n{{question}}\n\"\"\"\n【候选诊断】{{'\n' + options if options}}\n\n请直接写出最有可能的诊断的选项序号。",
    }
    instruction = ''
    evaluation_set = "test"
    load_args = ('uncertain_med/uncertain_med_bench', 'empec', 'rjua_disease', 'dialmed', 'cmb_exam', 'tcmsd_tcm', 'tcmsd_wm')
    example_set = None
    # load_args = ("/data/home/xiangxu_zhang/codes_repo/HealthLLM/benchmark/CMExam/data",)

    # def format_instance(self, instance):
    #     instance["target_idx"] = ord(instance["Answer"]) - ord('A')
    #     instance["options"] = [instance[op] for op in ("A", "B", "C", "D")]
    #     return instance

    def format_instance(self, instance):
        return dict(
            question=instance['question'],
            target_idx=ord(instance["answer"]) - ord('A'),
            options=instance['options'],
        )
    
    def _set_instruction(self, subset_name):
        self.instruction = self.instruction_dict[subset_name]

    def calculate_metric(self, predictions):# 选项结果[2, 2, 4, 2, 4, 4, 4, 2, 1, 2]
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @cached_property
    def references(self):
        # print([ord(instance["Answer"]) - ord('A') for instance in self.evaluation_data])
        return [ord(instance["answer"]) - ord('A') for instance in self.evaluation_data]
    
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        self._set_instruction(subset_name)
        self.evaluation_data = load_dataset(
            dataset_path,
            subset_name,
            split=evaluation_set,
            cache_dir=osp.join(dataset_path, '.cache'),
        )
            
