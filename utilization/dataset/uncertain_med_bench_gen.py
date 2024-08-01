from functools import cached_property
from logging import getLogger

from .generation_dataset import GenerationDataset
from datasets import load_dataset
from ..metric import Rouge, BERTScore
import os.path as osp
import random
# from ..metric import ECE, AUPRC, AUROC

logger = getLogger(__name__)

class UncertainMedBenchGen(GenerationDataset):
    instruction_dict = {
        'rjua_treatment': "你是一名医生，以下是一道医学治疗建议题，请你根据病情陈述进行诊断，并给出治疗建议，使用简洁的语言。\n\n【病情陈述】：\n\"\"\"\n{{question}}\n\"\"\"\n\n请使用简洁的语言进行诊断，并给出治疗建议。",
        'huatuo_encyclopedia': "请用简洁的语言回答下列医学问题。\n\n【问题】：\n\"\"\"\n{{question}}\n\"\"\"\n\n请用简洁的语言回答：",
        'qizhen_drug': "请用简洁的语言回答下列医学问题。\n\n【问题】：\n\"\"\"\n{{question}}\n\"\"\"\n\n请用简洁的语言回答：",
        'cmb_clin': "以下是一道病历分析题，请你基于病历回答问题。\n\n{{question}}\n\n请用简洁的语言回答：",
    }
    instruction = None
    evaluation_set = "test"
    metrics = [Rouge(), BERTScore()]
    # calib_metrics = [ECE(), AUROC(), AUPRC()]
    load_args = ('uncertain_med/uncertain_med_bench_gen', 'rjua_treatment', 'huatuo_encyclopedia', 'qizhen_drug', 'cmb_clin')
    example_set = None
    # load_args = ("/data/home/xiangxu_zhang/codes_repo/HealthLLM/benchmark/CMExam/data",)

    # def format_instance(self, instance):
    #     instance["target_idx"] = ord(instance["Answer"]) - ord('A')
    #     instance["options"] = [instance[op] for op in ("A", "B", "C", "D")]
    #     return instance

    def format_instance(self, instance):
        return dict(
            question=instance['question'],
            answer=instance['answer'],
        )
    
    def _set_instruction(self, subset_name):
        self.instruction = self.instruction_dict[subset_name]

    def calculate_metric(self, predictions):# 选项结果[2, 2, 4, 2, 4, 4, 4, 2, 1, 2]
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    def data_sample_with_threshold(self, subset_name, evaluation_data, local_rng, threshold=200):
        if len(evaluation_data) < threshold:
            logger.debug(f"{subset_name} indices: {[0, 1, 2, 3, 4]}")
            return evaluation_data
        else:
            sample_size = threshold
            # 使用随机抽样的方法从数据集中抽取样本
            # 设置局部种子
            sampled_indices = local_rng.sample(range(len(evaluation_data)), sample_size)
            logger.debug(f"{subset_name} indices: {sampled_indices[:5]}")
            # print(f"{subset_name} indices: {sampled_indices[:5]}")
            sampled_dataset = evaluation_data.select(sampled_indices)
            return sampled_dataset
        
    @cached_property
    def references(self):
        # print([ord(instance["Answer"]) - ord('A') for instance in self.evaluation_data])
        return [instance["answer"] for instance in self.evaluation_data]
    
    # def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
    #     if subset_name is None:
    #         subset_name = self.load_args[1]
    #     self._set_instruction(subset_name)
    #     self.evaluation_data = load_dataset(
    #         dataset_path,
    #         subset_name,
    #         split=evaluation_set,
    #         cache_dir=osp.join(dataset_path, '.cache'),
    #     )
    
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        self._set_instruction(subset_name)
        evaluation_data = load_dataset(
            dataset_path,
            subset_name,
            split=evaluation_set,
            cache_dir=osp.join(dataset_path, '.cache'),
        )
        local_rng = random.Random(subset_name)
        # TODO(xansar):增加threshold 参数
        self.evaluation_data = self.data_sample_with_threshold(subset_name, evaluation_data, local_rng)
    
            
    @cached_property
    def questions(self):
        return [instance["question"] for instance in self.evaluation_data]

    @cached_property
    def options(self):
        return ['No options' for instance in self.evaluation_data]
         