from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from sentence_transformers import SentenceTransformer

_LANGUAGES = ["zh"]

class Clue(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "CLUE",
            "hf_hub_name": "clue",
            "description": "CLUE, A Chinese Language Understanding Evaluation Benchmark",
            "reference": "https://aclanthology.org/2020.coling-main.419/",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": _LANGUAGES,
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 32,
        }

