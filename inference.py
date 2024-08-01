from utilization import get_evaluator, parse_argument
import os
import os.path as osp

def main():
    r"""The main pipeline for argument parsing, initialization, and evaluation."""
    model_args, dataset_args, evaluation_args = parse_argument(initalize=True)

    evaluator = get_evaluator(
        model_args=model_args,
        dataset_args=dataset_args,
        evaluation_args=evaluation_args,
        initalize=False,
    )
    evaluator.evaluate()

    # evaluation_args.uncertain_quantification = False
    # if evaluation_args.uncertain_quantification:
    #     dataset_args.dataset_names = ["uncertain_med_cons"]
    #     dataset_args.dataset_path = dataset_args.evaluation_results_path
    #     if not osp.exists('uncertain_results'):
    #         os.makedirs('uncertain_results')
    #     print(dataset_args.evaluation_results_path)
    #     dataset_args.evaluation_results_path = osp.join('uncertain_results', dataset_args.evaluation_results_path.split('/')[1])
    #     uncertain_evaluator = get_evaluator(
    #         model_args=model_args,
    #         dataset_args=dataset_args,
    #         evaluation_args=evaluation_args,
    #         initalize=False
    #     )
    #     uncertain_evaluator.evaluate()


if __name__ == "__main__":
    main()
