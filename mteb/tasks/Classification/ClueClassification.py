from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
import datasets
from mteb.evaluation.evaluators import (
    kNNClassificationEvaluator,
    kNNClassificationEvaluatorPytorch,
    logRegClassificationEvaluator,
)

import numpy as np

import logging
logger = logging.getLogger(__name__)

_LANGUAGES = ["zh"]


class TNEWS(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "TNEWS",
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

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], name='tnews', revision=self.description.get("revision", None)
        )
        self.data_loaded = True

    def _evaluate_monolingual(self, model, dataset, eval_split="test", train_split="train", **kwargs):
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]
        params = {"k": self.k, "batch_size": self.batch_size}
        params.update(kwargs)

        scores = []
        test_cache, idxs = None, None  # we store idxs to make the shuffling reproducible
        for i in range(self.n_experiments):
            logger.info("=" * 10 + f" Experiment {i+1}/{self.n_experiments} " + "=" * 10)
            # Bootstrap `self.samples_per_label` samples per label for each split
            X_sampled, y_sampled, idxs = self._undersample_data(
                train_split["sentence"], train_split["label"], self.samples_per_label, idxs
            )

            if self.method == "kNN":
                evaluator = kNNClassificationEvaluator(
                    X_sampled, y_sampled, eval_split["sentence"], eval_split["label"], **params
                )
            elif self.method == "kNN-pytorch":
                evaluator = kNNClassificationEvaluatorPytorch(
                    X_sampled, y_sampled, eval_split["sentence"], eval_split["label"], **params
                )
            elif self.method == "logReg":
                evaluator = logRegClassificationEvaluator(
                    X_sampled, y_sampled, eval_split["sentence"], eval_split["label"],max_iter=1000, **params
                )
            else:
                raise ValueError(f"Method {self.method} not supported")

            scores_exp, test_cache = evaluator(model, test_cache=test_cache)
            scores.append(scores_exp)

        if self.n_experiments == 1:
            return scores[0]
        else:
            avg_scores = {k: np.mean([s[k] for s in scores]) for k in scores[0].keys()}
            std_errors = {k + "_stderr": np.std([s[k] for s in scores]) for k in scores[0].keys()}
            return {**avg_scores, **std_errors}
