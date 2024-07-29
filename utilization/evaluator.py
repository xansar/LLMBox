from logging import getLogger
from typing import Any, Callable, Dict, List, Optional

from .load_dataset import load_datasets
from .load_model import load_model
from .utils.arguments import DatasetArguments, EvaluationArguments, ModelArguments, check_args
from .utils.catch_error import catch_error
from .utils.dynamic_stride_tqdm import dynamic_stride_tqdm
from .utils.log_results import PredictionWriter
from .utils.logging import log_once, set_logging
from .utils.random import set_seed

logger = getLogger(__name__)


class Evaluator:
    r"""The class for the evaluation pipeline.
    It loads the model and dataset, and then conducts evaluation.

    Args:
        args (Namespace): The global configurations.

    Attributes:
        model (Model): Our class for model.
        dataset (Dataset): Our class for dataset.
    """

    def __init__(
        self,
        *,
        model_args: ModelArguments,
        dataset_args: DatasetArguments,
        evaluation_args: Optional[EvaluationArguments] = None,
        initalize: bool = True,
        load_hf_model: Optional[Callable] = None,
        evaluation_data: Optional[List[Dict[str, Any]]] = None,
        example_data: Optional[List[Dict[str, Any]]] = None,
    ):

        self.model_args = model_args
        self.dataset_args = dataset_args
        evaluation_args = evaluation_args or EvaluationArguments()
        self.evaluation_args = evaluation_args
        if load_hf_model is not None:
            self.model_args.load_hf_model = load_hf_model

        if initalize:
            set_logging(self.model_args, self.dataset_args, self.evaluation_args)
            check_args(self.model_args, self.dataset_args, self.evaluation_args)
            logger.info(self.evaluation_args)

        set_seed(self.evaluation_args.seed)

        self.model = load_model(self.model_args)
        self.writer = PredictionWriter(self.dataset_args.evaluation_results_path)
        self.dataset = load_datasets(
            self.dataset_args,
            self.model,
            self.evaluation_args,
            evaluation_data=evaluation_data,
            example_data=example_data,
        )
        self.dataset.setup_metrics(self.model_args, self.dataset_args, self.evaluation_args)
        self.writer.write_metainfo(self.model_args, self.dataset_args, self.evaluation_args)

    @catch_error(True)
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        r"""It conducts the evaluation on the dataset with corresponding models.
        We support two evaluation types:

            - `Ranking`, ranking several options given a context, mainly applicable for multi-choice tasks. We compute the PPL scores of each option and select the one with lowest PPL.
            - `Generation`, generating the response based on the context, applicable for most of tasks. We directly call the `generation` interface of each model or API.

        Finally, we call the `calculate_metric` to get the metric score of prediction results.
        """
        from torch.utils.data import DataLoader

        if self.evaluation_args.dry_run:
            self.model.get_ppl = lambda x: [(0, 1)] * len(x)
            self.model.generation = lambda x: [""] * len(x)
            self.model.get_prob = lambda x: [[1 / p[1]] * p[1] for p in x]

        batch_sampler = self.dataset.get_batch_sampler(self.evaluation_args.dataloader_workers > 0)
        dataloader = DataLoader(
            self.dataset,
            collate_fn=lambda x: x,
            pin_memory=True,
            num_workers=self.evaluation_args.dataloader_workers,
            batch_sampler=batch_sampler,
        )
        call_model = batch_sampler.call_model

        # use tqdm for non-vllm models
        if self.dataset_args.batch_size != -1:
            # dataloader is often sacled by batch size and option nums, comparing to evaluation data
            dataloader = dynamic_stride_tqdm(
                dataloader,
                strides=self.dataset.strides,
                desc=self.dataset.name,
                dynamic_ncols=True,
                unit=" instances",
                continue_from=self.dataset_args.continue_from,
            )

        # call model
        raw_predictions = []
        if self.evaluation_args.continue_from:
            raw_predictions.extend(self.writer.load_continue())
        for batch in dataloader:
            log_once(logger.debug, f"batch_size {len(batch)}, first instance in batch:\n{batch[0]}", "fbi")
            batch_results = call_model(batch)
            if len(batch) != len(batch_results) and len(batch_results) != 0:
                raise RuntimeError(
                    f"The number of results {len(batch_results)} should be equal to the number of samples in the batch {len(batch)}."
                )
            if len(batch_results) > 0:
                log_once(logger.debug, f"first output in batch:\n{batch_results[0]}", "fbo")
            raw_predictions.extend(batch_results)
            self.dataset.step(self.writer, dataloader, batch_results)

        if self.evaluation_args.inference_only:
            logger.warning(
                f"Inference only mode, skip evaluation. Evaluate with flag `--continue_from {self.dataset_args.evaluation_results_path}`"
            )
            return {}
        if len(raw_predictions) != self.dataset.len():
            raise RuntimeError(
                f"The number of results {len(raw_predictions)} should be equal to the number of samples in the dataset {self.dataset.len()}."
            )

        # # post processing and self-consistency
        # predictions = self.dataset.post_processing(raw_predictions)
        # if len(predictions) != self.dataset.len(option_num=False, normalization=False):
        #     raise RuntimeError(
        #         f"The number of results {len(predictions)} should be equal to the number of samples in the dataset {self.dataset.len(option_num=False, normalization=False)}."
        #     )

        # step = self.dataset.len(option_num=False, sample_num=False, normalization=False)    # 问题数量
        # # TODO(xansar): 要写入参数文件
        # # self.evaluation_args.uncertain_quantification = True

        # #FIXME(xansar): 这里有问题,数据集是按照子集1,子集1,子集1,子集2,子集2,子集2排列的
        # if self.dataset_args.pass_at_k:
        #     mode_predictions = [predictions[i::step] for i in range(step)]
        # elif len(predictions) // step > 1:
        #     # TODO(xansar): 看看是否需要self-cons也跑一个
        #     if self.evaluation_args.uncertain_quantification:
        #         mode_predictions = [predictions[i::step][0] for i in range(step)]
        #     else:
        #         mode_predictions = [mode(predictions[i::step]) for i in range(step)]
        # else:
        #     mode_predictions = predictions

        # # calculate metric
        # metric_results, last_score_lists = self.dataset.calculate_metric(mode_predictions)

        # # calculate uncertainty
        # if self.evaluation_args.uncertain_quantification:
        #     from .uncertainty_quantification import SelfConsistency
        #     uncertainty_quan_func = SelfConsistency()
        #     calib_metric_results, calib_last_score_lists = \
        #         self.dataset.calculate_calibration_metric(
        #             predictions,
        #             last_score_lists,
        #             uncertainty_quan_func
        #             )

        #     for k in calib_metric_results.keys():
        #         metric_results[k].update(calib_metric_results[k])
        #     for i in range(len(calib_last_score_lists)):
        #         last_score_lists[i].update(calib_last_score_lists[i])



        #     # results = OrderedDict()
        #     # score_lists = []
        #     # grouped_display_names = defaultdict(list)  # group by dataset
        #     # splitted = self.dataset._split_by_subset(predictions, option_num=False, normalization=False, sample_num=self.dataset_args.sample_num)
        #     # for n, d, p, a in zip(self.dataset.display_names, self.dataset._datasets, splitted, last_score_lists):
        #     #     ## 计算uncertainty
        #     #     ### 按照sample_num对p进行折叠，构建一个二层list
        #     #     cur_step = d.len(option_num=False, sample_num=False, normalization=False)
        #     #     p_per_ques = [p[i::cur_step] for i in range(cur_step)]
        #     #     uncertainty_list = uncertainty_quan_func(p_per_ques)
        #     #     accuracy_list = a['Accuracy']
        #     #     subset_results, score_list = d.calculate_calibration_metric(uncertainty_list, accuracy_list)
        #     #     results.update(subset_results)
        #     #     score_lists.append(score_list)
        #     #     grouped_display_names[d.dataset_name].append(n)
            
        #     #         # calculate the mean of each category
        #     # for name, display_names in grouped_display_names.items():
        #     #     if self.categorized_subsets.get(name, None):
        #     #         for cat, cat_subsets in self.categorized_subsets[name].items():
        #     #             c = set(f"{name}:{s}" for s in cat_subsets)
        #     #             if len(c.intersection(set(display_names))) != len(c):
        #     #                 # skip if not all subsets of a category are available
        #     #                 continue
        #     #             fstr = f"{name}[{cat.title().replace('_', ' ')} Macro Average]"
        #     #             results[fstr] = avg_metrics([results[n] for n in c])

        #     # for k in results.keys():
        #     #     metric_results[k].update(results[k])
        #     # for i in range(len(score_lists)):
        #     #     last_score_lists[i].update(score_lists[i])
            

            
        #     # results[name + "[Marco Average]"] = avg_metrics([r for k, r in results.items() if k.startswith(name + ":")])

        # self.dataset.log_final_results(raw_predictions, predictions, mode_predictions, last_score_lists)
        # calculate metric
        metric_results = self.dataset.calculate_metric(raw_predictions)

        msg = f"Evaluation finished successfully:\nevaluation results: {self.dataset_args.evaluation_results_path}"
        for display_name, result in metric_results.items():
            if result is None:
                continue
            msg += f"\n##### {display_name} #####"
            for key, value in sorted(result.items(), key=lambda x: x[0]):
                msg += "\n{}: {:.2f}".format(key, value)

        logger.info(msg + "\n")
        return metric_results
