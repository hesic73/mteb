from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification

import datasets
from mteb.evaluation.evaluators import PairClassificationEvaluator
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

_LANGUAGES = ["zh"]


# # some of i
# _AVAILABLE_CONFIGS = ['afqmc', 'tnews', 'iflytek', 'cmnli', 'cluewsc2020', 'csl', 'cmrc2018', 'drcd', 'chid', 'c3', 'ocnli', 'diagnostics']


class AFQMCPairClassification(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "AFQMC",
            "hf_hub_name": "IDEA-CCNL/AFQMC",
            "description": "AFQMC",
            "reference": "https://github.com/IDEA-CCNL/Fengshenbang-LM/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["validation",
                            ],
            "eval_langs": _LANGUAGES,
            "main_score": "ap",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

      
        data_split = self.dataset[split]
        # tmp=[label for label in data_split["label"] if label not in [0,1]]
        # print(len(tmp))

        logging.getLogger("sentence_transformers.evaluation.PairClassificationEvaluator").setLevel(logging.WARN)
        evaluator = PairClassificationEvaluator(
            data_split["sentence1"], data_split["sentence2"], data_split["label"], **kwargs
        )
        scores = evaluator.compute_metrics(model)

        # Compute max
        max_scores = defaultdict(list)
        for sim_fct in scores:
            for metric in ["accuracy", "f1", "ap"]:
                max_scores[metric].append(scores[sim_fct][metric])

        for metric in max_scores:
            max_scores[metric] = max(max_scores[metric])

        scores["max"] = dict(max_scores)

        return scores
