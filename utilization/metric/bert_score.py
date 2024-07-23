import numpy as np
from bert_score import score

from .metric import Metric


class BERTScore(Metric):
    r"""Calculate the ROUGE score, including ROUGE_1, ROUGE_2, ROUGE_L

    Return:
        "ROUGE_1": float
        "ROUGE_2": float
        "ROUGE_L": float
    """
    def __init__(
            self, 
            verbose=True, 
            lang='zh',
            rescale_with_baseline=True,
            device=None, 
            batch_size=64, 
            use_fast_tokenizer=True, 
            *args, 
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.use_fast_tokenizer = use_fast_tokenizer


    def __call__(self, predictions, references):
        P, R, F1 = score(
            predictions, 
            references,
            model_type="/data/home/xiangxu_zhang/.model/google-bert/bert-base-chinese", 
            num_layers=8, 
            lang=self.lang,
            rescale_with_baseline=self.rescale_with_baseline,
            verbose=self.verbose,
            batch_size=self.batch_size,
            device=self.device,
            use_fast_tokenizer=self.use_fast_tokenizer,
            )

        self.last_score_lists = {"BERTScore": F1.tolist()}
        return {
            "BERTScore": F1.mean() * 100,
        }