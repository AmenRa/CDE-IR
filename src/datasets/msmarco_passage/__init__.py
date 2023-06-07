__all__ = [
    "download_msmarco",
    "EvalDataset",
    "TrainDataset",
]


from .download_utils import download_msmarco
from .eval_dataset import EvalDataset
from .paths import (
    dev_qrels_path,
    trec_dl_2019_qrels_path,
    trec_dl_2020_qrels_path,
    trec_dl_hard_qrels_path,
)
from .train_dataset import TrainDataset


def get_qrels_path(split: str) -> dict[str, dict[str, int]]:
    assert split in {"dev", "trec-dl-2019", "trec-dl-2020", "trec-dl-hard"}

    if split == "dev":
        return dev_qrels_path()
    elif split == "trec-dl-2019":
        return trec_dl_2019_qrels_path()
    elif split == "trec-dl-2020":
        return trec_dl_2020_qrels_path()
    elif split == "trec-dl-hard":
        return trec_dl_hard_qrels_path()
