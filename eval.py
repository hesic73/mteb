import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_DATASETS_CACHE'] = '/home/hesicheng/uniem/dataset'


from mteb import MTEB
from uniem import UniEmbedder
from argparse import Namespace
import argparse


ZH_TASKS = [
    "TNEWS",
    "AmazonReviewsClassification",
    "PAWS_X",
    "BQ",
    "LCQMC",
]


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/data/pretrained_models/uniem/m3e-base", type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    args=parser.parse_args()
    return args


def evaluate(uniem_model_name_or_path: str, output_dir: str):
    model = UniEmbedder.from_pretrained(uniem_model_name_or_path)

    for task in ZH_TASKS:
        evaluation = MTEB(tasks=[task], task_langs=['zh'])
        evaluation.run(model, verbosity=2, output_folder=os.path.join(output_dir,task))


if __name__ == "__main__":
    args=get_args()
    evaluate(args.model_name_or_path,args.output_dir)
