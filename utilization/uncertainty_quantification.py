from typing import Any, Callable, Dict, List, Optional
import numpy as np
from scipy.stats import entropy
import torch
import os
from tqdm import tqdm
import ast

import pandas as pd

from .utils import release_gpu_memory, release_vllm

def safe_extract_list_from_string(input_string, max_num=10):
    try:
        # 使用 ast.literal_eval 将字符串转换为列表
        result = ast.literal_eval(input_string)
        # 确认结果是一个列表
        if isinstance(result, list):
            # 限制最大要点数量,防止复读
            max_num = min(max_num, len(result))
            return result[:max_num]
        else:
            raise ValueError("The evaluated result is not a list")
    except (ValueError, SyntaxError) as e:
        print(f"Error while parsing the input string: {e}")
        # 处理异常的情况，可以返回一个默认值或者进一步处理
        return ['']
    

class ClaimExtractor:
    def __init__(self, 
                 model_path: str='/data/home/xiangxu_zhang/.model/Qwen/Qwen2-7B-Instruct',
                 gpu_memory_utilization: float=0.9
                 ):

        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm, self.tokenizer, self.sampling_params = self._load_llm_model()


    def _load_llm_model(self):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        # 模拟加载LLM模型的函数
        # 实际使用时应替换为实际的加载逻辑

        # # 根据实际情况设置tensor_parallel_size
        # tensor_parallel_size = os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") + 1

        # FIXME(xansar): vllm bug
        # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
        
        # llm = LLM(
        #     model="/data/home/xiangxu_zhang/.model/Qwen/Qwen2-7B-Instruct", 
        #     tensor_parallel_size=tensor_parallel_size,
        #     gpu_memory_utilization=self.gpu_memory_utilization,
        #     )
        
        llm = LLM(
            model="/data/home/xiangxu_zhang/.model/Qwen/Qwen2-7B-Instruct",
            tokenizer="/data/home/xiangxu_zhang/.model/Qwen/Qwen2-7B-Instruct",
            tensor_parallel_size=torch.cuda.device_count(),
            # dtype=args.torch_dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            # quantization="gptq" if args.gptq else None,
            trust_remote_code=True,
            # seed=args.seed,
            max_logprobs=40,  # https://github.com/vllm-project/vllm/issues/5299
            # **kwargs
        )  # type: ignore
        tokenizer = llm.get_tokenizer()
        tokenizer.truncation_side = "left"
        tokenizer.model_max_length = min(
            llm.llm_engine.model_config.max_model_len,
            512
        )
        if hasattr(tokenizer, "add_bos_token"):
            # add in chat_template
            setattr(tokenizer, "add_bos_token", False)
        if hasattr(tokenizer, "add_eos_token"):
            setattr(tokenizer, "add_eos_token", False)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        sampling_params = SamplingParams(temperature=0, max_tokens=512)
        
        return llm, tokenizer, sampling_params

    def _extract_keypoints(self, primary_answers: List[str], questions: List[str]) -> List[List[str]]:
        """
        提取答案的要点
        """
        prompt = ("根据问题,将回答整理为要点,每个要点仅包含一个事实声明.返回python列表格式\n"
                  "----------------------\n"
                  "问题:\n"
                  "{question}\n"
                  "回答:\n"
                  "{answer}\n"
                  "----------------------\n"
                  "请将回答整理为要点,每个要点仅包含一个事实声明.\n")
        messages = [
            [
                {"role": "system", "content": "完成指令,返回python列表格式"},
                {"role": "user", "content": prompt.format(question=question, answer=answer)}
            ]
            for question, answer in zip(questions, primary_answers)
        ]
        texts = [
                self.tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                ) for msg in messages
        ]
        # 提取要点
        outputs = self.llm.generate(texts, self.sampling_params)

        # from .utils import release_vllm
        release_vllm(self.llm)
        return outputs

class EntailDataset(torch.utils.data.Dataset):
    def __init__(self, total_instances: List[Dict[str, torch.Tensor]], question_idx_lst: List[int], claim_idx_lst: List[int]):
        self.total_instances = total_instances
        self.question_idx_lst = question_idx_lst
        self.claim_idx_lst = claim_idx_lst

    @staticmethod
    def _format(claim: str, answer: str, question: str, tokenizer: Callable, max_len: int=512) -> Dict[str, torch.Tensor]:
        """
        计算claim相似度

        answer + claim >= 512
            - 截断answer
        answer + claim < 512
            - 截断question填充
        """
        base_len = len(tokenizer.encode(answer, claim))
        middle_string = ' (...中间内容省略...) '
        if base_len > max_len:
            answer_res_len = max_len - len(tokenizer.encode(claim, middle_string)) - 1
            answer_tokens = tokenizer.encode(
                answer
                )
            left = tokenizer.decode(
                answer_tokens[:answer_res_len//2],
                skip_special_tokens=True,
                spaces_between_special_tokens=False
                ).replace(' ', '')
            right = tokenizer.decode(
                answer_tokens[-answer_res_len//2:],
                skip_special_tokens=True,
                spaces_between_special_tokens=False
                ).replace(' ', '')
            answer_omit = left + middle_string + right
            question_omit = ''
        else:
            # 限制长度
            res_len = max_len - base_len
            
            if res_len >= len(tokenizer.encode(question)):
            # 如果res_len大于question的长度,则直接拼
                question_omit = question
            elif res_len <= len(tokenizer.encode(middle_string)):
                question_omit = ''
            else:
                res_len = max_len - base_len - len(tokenizer.encode(middle_string))
                # 从question中补足限制长度的内容,首尾截取,保证拼接后长度正好为max_len
                question_tokens = tokenizer.encode(
                    answer
                    )
                ques_left = tokenizer.decode(
                    question_tokens[:res_len//2],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False
                    ).replace(' ', '')
                ques_right = tokenizer.decode(
                    question_tokens[-res_len//2:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False
                    ).replace(' ', '')
                question_omit = ques_left + middle_string + ques_right
            
            answer_omit = answer
        
        texta = question_omit + ' ' + answer_omit # 前提
        textb = claim  # 假设
        return tokenizer.encode_plus(texta, textb, max_length=512, truncation=True,padding='max_length',return_tensors='pt')
    
    @classmethod
    def _load_data(
        cls, 
        claims_seq: List[List[str]], 
        other_answers_seq: List[List[str]], 
        questions: List[str],
        tokenizer: Callable,
        ):
        total_instances = []
        question_idx_lst = []
        claim_idx_lst = []
        tqdm_bar = tqdm(total=sum([len(item) for item in claims_seq]), desc="Constructing entailment instances")
        for ques_id, (claims, oth_ans, ques) in enumerate(zip(claims_seq, other_answers_seq, questions)):
            for claim_id, claim in enumerate(claims):
                for answer in oth_ans:
                    ins = cls._format(claim, answer, ques, tokenizer)  # {'input_ids': tensor, 'attention_mask': tensor}  
                    total_instances.append(ins)
                    question_idx_lst.append(ques_id)
                    claim_idx_lst.append(claim_id)
                tqdm_bar.update(1)
        return cls(total_instances, question_idx_lst, claim_idx_lst)

    def __len__(self):
        return len(self.total_instances)

    def __getitem__(self, idx):
        return self.total_instances[idx]
    

class ClaimEntailScore:
    def __init__(self, 
                 model_path: str='/data/home/xiangxu_zhang/.model/IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI',
                 device: str='cuda',
                 batch_size: int=16
                 ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

        self.tokenizer, self.model = self._load_nli_model()
    
    def _load_nli_model(self):
        from transformers import BertTokenizer, AutoModelForSequenceClassification
        # 模拟加载NLI模型的函数
        # 实际使用时应替换为实际的加载逻辑
        tokenizer=BertTokenizer.from_pretrained(
            self.model_path)
        model=AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            ).to(self.device)
        return tokenizer, model

    # def _format(self, claim: str, answer: str, question: str) -> float:
    #     """
    #     计算claim相似度
    #     """
    #     max_len = 512
    #     # 基本长度是claim和answer的长度
    #     """
    #     answer + claim >= 512
    #         - 截断answer
    #     answer + claim < 512
    #         - 截断question填充
    #     """
    #     base_len = len(self.tokenizer.encode(answer, claim))
    #     middle_string = ' (...中间内容省略...) '
    #     if base_len > max_len:
    #         answer_res_len = max_len - len(self.tokenizer.encode(claim, middle_string)) - 1
    #         answer_tokens = self.tokenizer.encode(
    #             answer
    #             )
    #         left = self.tokenizer.decode(
    #             answer_tokens[:answer_res_len//2],
    #             skip_special_tokens=True,
    #             spaces_between_special_tokens=False
    #             ).replace(' ', '')
    #         right = self.tokenizer.decode(
    #             answer_tokens[-answer_res_len//2:],
    #             skip_special_tokens=True,
    #             spaces_between_special_tokens=False
    #             ).replace(' ', '')
    #         answer_omit = left + middle_string + right
    #         question_omit = ''
    #     else:
    #         # 限制长度
    #         res_len = max_len - base_len
            
    #         if res_len >= len(self.tokenizer.encode(question)):
    #         # 如果res_len大于question的长度,则直接拼
    #             question_omit = question
    #         elif res_len <= len(self.tokenizer.encode(middle_string)):
    #             question_omit = ''
    #         else:
    #             res_len = max_len - base_len - len(self.tokenizer.encode(middle_string))
    #             # 从question中补足限制长度的内容,首尾截取,保证拼接后长度正好为max_len
    #             question_tokens = self.tokenizer.encode(
    #                 answer
    #                 )
    #             ques_left = self.tokenizer.decode(
    #                 question_tokens[:res_len//2],
    #                 skip_special_tokens=True,
    #                 spaces_between_special_tokens=False
    #                 ).replace(' ', '')
    #             ques_right = self.tokenizer.decode(
    #                 question_tokens[-res_len//2:],
    #                 skip_special_tokens=True,
    #                 spaces_between_special_tokens=False
    #                 ).replace(' ', '')
    #             question_omit = ques_left + middle_string + ques_right
            
    #         answer_omit = answer
        
    #     texta = question_omit + ' ' + answer_omit # 前提
    #     textb = claim  # 假设
    #     return self.tokenizer.encode_plus(texta, textb, max_length=512, truncation=True,padding='max_length',return_tensors='pt')
    #     # output = self.model(torch.tensor([self.tokenizer.encode(texta,textb)]).to(self.device))
    #     # entail_prob = torch.nn.functional.softmax(output.logits,dim=-1)[0, -1].detach().cpu().item()    # 蕴含概率
    #     # return entail_prob
    
    def __call__(self, claims_seq: List[List[str]], other_answers_seq: List[List[str]], questions: List[str]) -> List[float]:
        """
        计算claim与other_answer的相似度
        """
        # total_pairs = []
        # question_idx_lst = []
        # claim_idx_lst = []
        # for ques_id, (claims, oth_ans, ques) in enumerate(zip(claims_seq, other_answers_seq, questions)):
        #     for claim_id, claim in enumerate(claims):
        #         for answer in oth_ans:
        #             ins = self._format(claim, answer, ques)
        #             total_pairs.append(ins)
        #             question_idx_lst.append(ques_id)
        #             claim_idx_lst.append(claim_id)

        # 批量推理
        # FIXME(xansar): 改一下dataset接口和数据处理
        dataset = EntailDataset._load_data(claims_seq, other_answers_seq, questions, self.tokenizer)
        def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            input_ids = torch.vstack([x['input_ids'] for x in batch])
            attention_mask = torch.vstack([x['attention_mask'] for x in batch])
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
            collate_fn=_collate_fn)
        
        self.model.eval()
        total_results = []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc="Calculating entailment score"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                entail_prob = torch.nn.functional.softmax(outputs.logits,dim=-1)[:, -1].detach().cpu().tolist()    # 蕴含概率
                total_results.extend(entail_prob)

        # 计算蕴含概率
        df_results = pd.DataFrame({
            'ques_id': dataset.question_idx_lst, 
            'claim_id': dataset.claim_idx_lst, 
            'res': total_results
            })

        # reverse map 根据question_idx_lst, claim_idx_lst 计算每个claim的相似度
        avg_claim_level = df_results.groupby(['ques_id', 'claim_id'])['res'].mean().reset_index()

        avg_ques_level = avg_claim_level.groupby('ques_id')['res'].mean().reset_index()

        sims = avg_ques_level['res'].tolist()
        return sims

        
class SelfConsistency: 
    # def _calculate_degree_confidence(self, prediction: List[int]) -> float:
    #     """
    #     参考https://github.com/zlin7/UQ-NLG
    #     Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models

    #     利用response的多样性来计算置信度
    #     Conf(x, s_j) = D_jj / m, s_j是主生成内容, D_jj等于s_j和其他生成内容的相似度之和, m是生成内容的数量

    #     Args:
    #         prediction: List[int], 模型的预测结果 
        
    #     Returns:
    #         float, 置信度
    #     """
    #     # 假设第一个答案就是主要答案
    #     # TODO(xansar): 可以考虑使用其他答案作为主要答案
    #     primary_answer = prediction[0]
    #     other_ansers = prediction[1:]

    #     # 计算主要答案与其余答案的相似度之和(不包含自身)
    #     similarity_sum = sum([1 for answer in prediction[1:] if answer == primary_answer])

    #     # 计算置信度
    #     return similarity_sum / len(prediction) - 1
    def __init__(self, is_gen: bool=False, gpu_memory_utilization: float=0.9) -> None:
        self.is_gen = is_gen
        self.gpu_memory_utilization = gpu_memory_utilization

    def _conf_mcqa(self, primary_answer_seq: List[int], other_answers_seq: List[List[int]]) -> List[float]:
        def _conf(primary_answer, other_answers):
            # 计算主要答案与其余答案的相似度之和(不包含自身)
            return np.mean([1 for answer in other_answers if answer == primary_answer])
        
        return [_conf(pri_ans, oth_ans) 
                     for pri_ans, oth_ans in zip(primary_answer_seq, other_answers_seq)]

    def _conf_gen(
            self, 
            claims_seq: List[List[str]], 
            other_answers_seq: List[List[str]], 
            questions: List[str],
            ) -> List[float]:
        """
        参考
        [arxiv'24] Calibrating Long-form Generations from Large Language Models
        [arxiv'24] LUQ: Long-text Uncertainty Quantification for LLMs
        从主要答案中提取要点, 并计算其他答案与主要答案的蕴含概率作为相似度

        """
        entail_score = ClaimEntailScore()
        confs = entail_score(claims_seq, other_answers_seq, questions)

        del entail_score
        release_gpu_memory()
        return confs
        # tqdm_bar = tqdm(total=len(claims_seq), desc="Calculating entailment score")

        # # 构建pipeline
        # def _conf(claims, other_answers, question):
        #     sim_lst = []
        #     for claim in claims:
        #         # 计算当前claim与其他答案的相似度
        #         similarity_sum = entail_score(claim, other_answers, question)
        #         sim_lst.append(similarity_sum)
        #     tqdm_bar.update(1)
        #     return np.mean(sim_lst)
        # # 加载NLI模型
        # entail_score = self.entail_score
        # confs = [_conf(claims, oth_ans, ques) 
        #              for claims, oth_ans, ques in zip(claims_seq, other_answers_seq, questions)]
        # return confs
        

    def _extract_keypoints(
            self, 
            primary_answers: List[str], 
            questions: List[str],
            ) -> List[List[str]]:
        """
        提取答案的要点
        """
        claim_extractor = ClaimExtractor(gpu_memory_utilization=self.gpu_memory_utilization)
        keypoints = claim_extractor._extract_keypoints(primary_answers, questions)
        claims_seq = [
            safe_extract_list_from_string(claim.outputs[0].text, max_num=10)  # 限制最多10个要点
            for claim in keypoints
        ]
        return claims_seq

    def __call__(
            self, 
            predictions: List[List[Any]], 
            questions: List[str], 
            ) -> List[float]:

        primary_answer_seq = [prediction[0] for prediction in predictions]
        other_answers_seq = [prediction[1:] for prediction in predictions]
        if self.is_gen:
            # 生成式,走claim相似度
            ## 批量生成claim
            claims_seq = self._extract_keypoints(primary_answer_seq, questions)
            ## 计算claim相似度
            confs = self._conf_gen(claims_seq, other_answers_seq, questions)
        else:
            # mcqa,走匹配相似度
            confs = self._conf_mcqa(primary_answer_seq, other_answers_seq)
            claims_seq = [None] * len(primary_answer_seq)
        return confs, claims_seq
        
        