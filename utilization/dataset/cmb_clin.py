from functools import cached_property

from ..metric import Rouge
from .generation_dataset import GenerationDataset
from .dataset_utils import load_raw_dataset_from_file, get_raw_dataset_loader
import os.path as osp

class CMBClin(GenerationDataset):
    """The dataset of tl;dr

    The TL;DR Dataset is an English-language dataset containing
    Reddit posts to summarize.

    Examples:
        prompt: SUBREDDIT: r/loseit TITLE: SV & NSV! Keeping on keeping on. POST: 30F, 5'6". SW: 236 GW: 150 CW: 219 I weigh myself weekly and measure myself monthly. I'd hit a plateau the last four weeks or so where I was stuck at 222. Felt like kind of a bummer, but knew it's because I haven't been as strict as I should with my diet, and the last week and a half have been crazy with life things, so I haven't been exercising as frequently as I've gotten used to. When I weighed myself as normal on Monday, I was kind of disappointed to see the scale not budging and figured it was time to buckle down again and really watch my diet. Today was my measure-in day, and I've felt cruddy in general since Monday because I caught some chest congestion/cold bug over the weekend. I get on the scale...it says 219. Whaaaaat? I take my measurements, which are down slightly from last month, and with an total-body loss of 8 inches from my starting point on 12/23/14! Some of my clothes have been feeling a bit looser as of late and now I know it's just not in my head. I'm now the lightest and smallest I've been since right around high school! TL;DR:
        label: Progress is still happening, even when you think it might not be! Don't get discouraged, even if your journey seems to be going slowly. Don't give up, warriors.
    """

    instruction = "以下是一位病人的病例：\n{description}\n问题：{question}\n"
    evaluation_set = "test"
    example_set = None
    metrics = [Rouge()]

    def format_instance(self, instance):
        return dict(
            description=instance['description'],
            question=instance['question'],
            target=instance['answer'],
            )

    @cached_property
    def references(self):
        return [instance["answer"] for instance in self.evaluation_data]
    
    def preprocess_dataset(self, raw_dataset):
        preprocessed_dataset = []
        for instance in raw_dataset:
            for sub_instance in instance['QA_pairs']:
                question = sub_instance['question']
                answer = sub_instance['answer']
                preprocessed_dataset.append({
                    'title': instance['title'].strip().strip('\n'),
                    'description': instance['description'].strip().strip('\n'),
                    'question': question.strip().strip('\n'),
                    'answer': answer.strip().strip('\n'),
                })
                
        return preprocessed_dataset
    
    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        # print(dataset_path, subset_name, evaluation_set, example_set)

        # example_path = osp.join(dataset_path, example_set + '.jsonl')
        if evaluation_set == 'test':
            evaluation_path = osp.join(dataset_path, 'CMB-Clin-qa.json')
        else:
            raise ValueError(f"Unknown evaluation set: {self.evaluation_set}")
        self.evaluation_data = load_raw_dataset_from_file(evaluation_path)
        self.evaluation_data = self.preprocess_dataset(self.evaluation_data)
        # self.example_data = load_raw_dataset_from_file(example_path)
        # self.example_data = self.preprocess_dataset(self.example_data)