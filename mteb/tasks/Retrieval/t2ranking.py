from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
import sys
import csv
csv.field_size_limit(sys.maxsize)


import datasets

import logging
from time import time
from typing import Dict, List

from mteb.abstasks.AbsTaskRetrieval import DRESModel

logger = logging.getLogger(__name__)

class T2RANKING(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "T2RANKING",
            "hf_hub_name":  "THUIR/T2Ranking",
            "description": "THUIR/T2Ranking",
            "reference": "https://huggingface.co/datasets/THUIR/T2Ranking",
            "description": "THUIR/T2Ranking",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
        }
        
    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
            
        # 参考 beir.datasets.data_loader.GenericDataLoader

        # TODO: add split argument
        self.corpus = datasets.load_dataset(
            self.description["hf_hub_name"],name="collection", revision=self.description.get("revision", None)
        # ,use_auth_token="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"
        )
        self.corpus={str(d['pid']):{"text":d['text'],"title":""} for d in self.corpus['train']}
        
        self.queries=datasets.load_dataset(
            self.description["hf_hub_name"],name="queries.test", revision=self.description.get("revision", None)
        )
        self.queries={query['qid']:query['text'] for query in self.queries['train']}

        self.relevant_docs =datasets.load_dataset(
            self.description["hf_hub_name"],name="qrels.dev", revision=self.description.get("revision", None)
        )
        self.relevant_docs={str(doc['qid']):{str(doc['pid']):doc["rel"] } for doc in self.relevant_docs['train']}
        print(len(self.corpus),len(self.queries),len(self.relevant_docs))

        self.data_loaded = True
    
    def evaluate(
        self,
        model,
        split="test",
        batch_size=128,
        corpus_chunk_size=None,
        target_devices=None,
        score_function="cos_sim",
        **kwargs
    ):
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus, self.queries, self.relevant_docs


        if target_devices is not None:
            logger.warning(
                    "DenseRetrievalParallelExactSearch could not be imported from beir. Using DenseRetrievalExactSearch instead."
                )
            logger.warning("The parameter target_devices is ignored.")

        from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

        model = model if self.is_dres_compatible(model, is_parallel=False) else DRESModel(model)

        model = DRES(
                model,
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 50000,
                **kwargs,
            )

        retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim" or "dot"
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values)
        mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }

        return scores



