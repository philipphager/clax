import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict

import numpy as np
import pandas as pd

from progress_table import ProgressTable
from pyclick.click_models.CTR import RCTR, DCTR, GCTR
from pyclick.click_models import PBM, UBM, DBN, DCM, CCM

from pyclick.click_models.task_centric.TaskCentricSearchSession import (
    TaskCentricSearchSession,
)
from pyclick.search_session import SearchResult
from tqdm import tqdm

from clix.datasets.yandex import YandexDataset
from clix.metrics import MultiMetric, LogLikelihood, Perplexity, ConditionalPerplexity
from clix.models.math import probs_to_log_probs


class PyClickTrainer:

    def train(self, model, train_dataset: YandexDataset):
        timer_start = perf_counter()

        sessions = tqdm(
            [self._to_session(row) for row in train_dataset],
            desc="Creating PyClick sessions...",
        )
        print("Begin training...")
        model.train(sessions)

        timer_stop = perf_counter()
        time_elapsed = timer_stop - timer_start
        print(
            f"Training complete in: {datetime.timedelta(seconds=time_elapsed)} or "
            f"{time_elapsed} seconds"
        )

    def test(self, model, dataset: YandexDataset) -> Dict:
        metrics = MultiMetric(
            **{
                "ll": LogLikelihood(),
                "ppl": Perplexity(),
                "cond_ppl": ConditionalPerplexity(),
            }
        )

        logger = ProgressTable(
            columns=[
                "model",
                *metrics.compute(prefix="test_").keys(),
            ],
            pbar_embedded=False,
            pbar_show_percents=True,
            pbar_style="angled alt red blue",
        )
        logger.update("model", type(model).__name__)

        for row in tqdm(dataset, desc="Test"):
            session = self._to_session(row)
            probs = np.array(model.get_full_click_probs(session))
            conditional_probs = np.array(model.get_conditional_click_probs(session))

            # PyClick already inverts predictions based on clicks. Invert, so that
            # we consistently predict P(C = 1 |...):
            conditional_probs = np.where(
                row["clicks"],
                conditional_probs,
                1 - conditional_probs,
            )

            # Clax uses log probs:
            log_probs = probs_to_log_probs(probs)
            conditional_log_probs = probs_to_log_probs(conditional_probs)

            metrics.update(
                log_probs=log_probs.reshape(1, -1),
                conditional_log_probs=conditional_log_probs.reshape(1, -1),
                clicks=row["clicks"].reshape(1, -1),
                where=row["mask"].reshape(1, -1),
            )

        results = metrics.compute("test_")
        logger.update_from_dict(results)
        logger.close()
        return logger.to_df()

    @staticmethod
    def _to_session(row: Dict[str, np.ndarray]) -> TaskCentricSearchSession:
        query = str(int(row["query_id"]))
        session = TaskCentricSearchSession(task="0", query=query)
        results = []

        for i in range(row["n"]):
            result = SearchResult(
                search_result_id=str(row["query_doc_ids"][i]),
                click=int(row["clicks"][i]),
            )
            results.append(result)

        session.web_results = results
        return session


def main():
    path = Path("/ivi/ilps/datasets/yandex/relevance_prediction/YandexClicks.txt")
    index_path = Path("data/wscd-2012/index.json")

    train_dataset = YandexDataset(
        path,
        index_path,
        session_range=(0, 600_000),
    )
    test_dataset = YandexDataset(
        path,
        index_path,
        session_range=(800_000, 1_000_000),
    )

    models = [GCTR(), RCTR(), DCTR(), PBM(), UBM(), DBN(), CCM(), DCM()]

    test_dfs = []

    for model in models:
        trainer = PyClickTrainer()
        trainer.train(model, train_dataset)
        test_df = trainer.test(model, test_dataset)
        test_dfs.append(test_df)

        pd.concat(test_dfs).to_csv("em_test.csv")


if __name__ == "__main__":
    main()
